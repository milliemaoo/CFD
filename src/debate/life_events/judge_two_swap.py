"""LLM-as-a-judge over life_events debates.

Reads the debate transcripts at --questions-path, renders both agents' rounds for a single judge model
and asks it for the final label set. --conf-render picks how the agents'
confidence reaches the judge:

- "scalar" — one score per round, as a bare number, which is how the
  coarse-grained entropy driver stores it.
- "coarse" — the same rendering for a score that arrives nested in a dict
  under `final_single_conf_1_10`, as the sampling and post-hoc drivers store
  it.
- "fine" — one score per category, nested in a dict under
  `raw_exp_confidences_1_10` and `filtered_final_confidence_1_10`, which is how
  the drivers store it under `--conf-method fg_confidence`. The response is
  re-parsed and rendered with `(Confidence: X)` after each of the 8 life-event
  categories and after each selected answer label.

Which agent is shown first is a confound, so each example is rendered twice —
original order and with the two agents swapped — and one of the two is chosen
per example by a `random.Random(seed)` draw. Writes
{question_id: (completion, prompt_used, gold)} to --output-path.

--questions-path and --output-path may contain `{seed}`, which is filled in
per seed.
"""

import json
import random
import argparse

from debate.gen_utils import (
    generate_answer_standard,
    load_few_shot,
)
from debate.life_events.parsing import parse_formatted_response
from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer
from debate.seeding import set_seed

def swap_agents_in_example(example):
    new_example = dict(example)  # shallow copy of the outer dict
    agents = example["agents"]
    swapped_agents = {
        "agent_0": agents["agent_1"],
        "agent_1": agents["agent_0"],
    }
    new_example["agents"] = swapped_agents
    return new_example

_FEW_SHOT_EXAMPLES = load_few_shot("life_events")


def _render_prompt(question_id, formatted_agent_blocks):
    """Wrap one rendered transcript in the shared judge instruction."""
    few_shot_examples = _FEW_SHOT_EXAMPLES
    prompt_template = f"""You are a judge tasked with evaluating a debate between two agents about whether a given post reflects one or more predefined life event categories.

For each category, analyze and compare the agents' arguments for accuracy and relevance. Judge whether their reasoning is well supported by the post content and whether it aligns with the given category definitions.

When comparing agents, treat confidence scores as secondary signals. Base your decision primarily on the strength and clarity of evidence in the post. High confidence without explicit textual support should be treated as weak reasoning.

Then, based on this comparison, decide whether the category applies and explain your reasoning.

---

Life Event Definition:

Life events are experiences that have a major personal impact on an individual. They must involve a clearly identifiable occurrence or change. These events may have occurred in the past (explicitly stated, or inferred from context if the impact is clear), be occurring in the present (explicitly described or clearly implied), or be expected in the near future (only if explicitly stated).

- Mental Health: 
Mental health life events cover discussions about receiving a formal mental health diagnosis; starting or adjusting psychiatric medication; beginning therapy; recovery or significant symptom improvement; or experiencing acute psychological episodes such as manic episodes, psychotic breaks, panic attacks, dissociative episodes, self-harm, suicide attempts, or acute suicidal ideation requiring crisis intervention. It does not include general low mood, stress, anxiety, or depression unless explicitly tied to one of these major events.
- Physical Health: 
Physical health life events cover accidents; injuries; diagnoses or survival of serious illnesses; chronic conditions that have a notable impact on daily life; or cases where the individual or someone important to them has been hospitalized, undergone surgery, received emergency medical treatment, or experienced other major medical interventions. Pregnancy-related experiences, including becoming pregnant, pregnancy loss, abortion, or complications, are also included. Common aches, headaches, fatigue, or other symptoms without being part of a major physical health event are not included.
- Abuse & Addiction: 
Covers major life events involving abuse or substance-related issues. Abuse may occur in childhood or adulthood and may take physical, sexual, emotional, psychological, or other forms. Addiction-related events include the onset of heavy drug or alcohol use; acknowledgment of addiction; overdoses; withdrawal symptoms; relapses; or recovery (including participation in treatment programs such as rehabilitation or support groups).
- Relationship & Loss: 
Covers specific relationship changes or disruptions in meaningful personal connections—family or non-family—that result in strong emotional impact. Includes beginning or ending friendships or romantic relationships; marriage or divorce; parental separation or divorce; serious arguments or conflicts; relationships becoming abusive or toxic; the addition of new family members (e.g., through birth or adoption); parenting difficulties; emotionally distressing health issues involving family members, partners, close friends, or other meaningful connections; or the death of a person or pet who was important to the individual.
- Career & Education: 
Career-related life events cover starting or losing a job; promotions; demotions; troubles at work; being unable to find a job; retirement; or becoming a business owner. Education-related life events include starting or graduating from school; transferring schools; leaving school without graduating; being denied entry into school; taking major exams; experiencing significant academic challenges; or achieving major academic milestones (e.g., earning a certification).
- Financial & Legal & Societal: 
Financial life events include major purchases (e.g., a house or car); financial gains or milestones (e.g., paying off debt); financial losses (e.g., stolen or damaged property); or ongoing financial difficulties. Legal life events include law violations; arrests; court appearances; lawsuits or legal actions; or major interactions with the justice system, whether the individual is subject to legal proceedings or initiating them. Societal life events with clear personal impact include natural disasters; pandemics; major political events; societal issues; or notable public encounters (e.g., meeting a celebrity).
- Lifestyle & Identity & Environment: 
Covers major life events involving changes in lifestyle habits, personal identity, or living environment. Lifestyle changes include adopting new health- and well-being-related routines, such as adjustments in diet; exercise; sleep management; stress management (e.g., through meditation); reducing substance use; fostering positive social connections; or getting a pet. It does not include clinical treatment for mental health issues. Identity transitions include identifying or changing one's gender or sexual orientation (e.g., coming out as LGBTQ+); discussing a new sexual experience; or exploring, adopting, or changing political or religious beliefs (including changes in religious practices). Relocation life events include moving to a new place; moving in or out of the family home; family members moving into or out of the household; or significant travel experiences.
- None: 
Applies when the post does not clearly reflect any of the above life event categories.

---

Below are some examples:
{few_shot_examples}

Post to Analyze:
"{question_id}"

Transcript of the Debate:
{formatted_agent_blocks}
---

Please follow the exact format below.

Output Format:
Explanation:
- Mental Health: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Physical Health: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Abuse & Addiction: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Relationship & Loss: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Career & Education: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Financial & Legal & Societal: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Lifestyle & Identity & Environment: [Compare, explain, and conclude]. So the answer is yes (or is no).
- None: [Compare, explain, and conclude]. So the answer is yes (or is no).

Answer:
Output exact category names: Mental Health, Physical Health, Abuse & Addiction, Relationship & Loss, Career & Education, Financial & Legal & Societal, Lifestyle & Identity & Environment, or None. Use commas to separate multiple labels, if any."""
    return prompt_template

def _judge_prompt_scalar(example):
    question_id = example["question_id"]
    agents = example["agents"]

    formatted_agent_blocks = ""
    for agent_id, agent_info in agents.items():
        responses = agent_info["responses"][:2]
        confidences = agent_info.get("confidences", [])

        formatted_agent_blocks += f"\n{agent_id}:\n"

        for round_idx, resp in enumerate(responses):
            formatted_agent_blocks += f"\nRound {round_idx + 1}:\n"
            formatted_agent_blocks += resp.strip() + "\n\n"

            final_single_conf = None
            if round_idx < len(confidences):
                final_single_conf = confidences[round_idx]

            formatted_agent_blocks += f"Confidence: {final_single_conf if final_single_conf is not None else 'N/A'}\n"

    return _render_prompt(question_id, formatted_agent_blocks)

def _judge_prompt_coarse(example):
    question_id = example["question_id"]
    agents = example["agents"]

    formatted_agent_blocks = ""
    for agent_id, agent_info in agents.items():
        responses = agent_info["responses"][:2]
        confidences = agent_info.get("confidences", [])

        formatted_agent_blocks += f"\n{agent_id}:\n"

        for round_idx, resp in enumerate(responses):
            formatted_agent_blocks += f"\nRound {round_idx + 1}:\n"
            formatted_agent_blocks += resp.strip() + "\n\n"

            # take final_single_conf_1_10 only
            final_single_conf = None
            if round_idx < len(confidences):
                final_single_conf = confidences[round_idx].get("final_single_conf_1_10")

            formatted_agent_blocks += f"Confidence: {final_single_conf if final_single_conf is not None else 'N/A'}\n"

    return _render_prompt(question_id, formatted_agent_blocks)

def _judge_prompt_fine(example):
    question_id = example["question_id"]
    agents = example["agents"]

    all_labels = [
        "Mental Health", "Physical Health", "Abuse & Addiction",
        "Relationship & Loss", "Career & Education",
        "Financial & Legal & Societal", "Lifestyle & Identity & Environment", "None"
    ]

    formatted_agent_blocks = ""
    for agent_id, agent_info in agents.items():
        responses = agent_info["responses"][:2]
        confidences = agent_info.get("confidences", [])

        formatted_agent_blocks += f"\n{agent_id}:\n"

        for i, response in enumerate(responses):
            formatted_agent_blocks += f"\nRound {i+1}:\n"
            parsed = parse_formatted_response(response)

            raw_conf = confidences[i].get("raw_exp_confidences_1_10", {}) if i < len(confidences) else {}
            final_conf = confidences[i].get("filtered_final_confidence_1_10", {}) if i < len(confidences) else {}

            # Render explanation plus confidence
            formatted_agent_blocks += "Explanation:\n"
            for cat in all_labels:
                explanation = parsed["explanations"].get(cat, "No explanation found.")
                conf = raw_conf.get(cat)
                conf_str = f" (Confidence: {conf})" if conf is not None else ""
                formatted_agent_blocks += f"- {cat}: {explanation}{conf_str}\n"

            # Render the answer
            formatted_agent_blocks += "\nAnswer:\n"
            answers = []
            for cat in parsed["answer"]:
                conf = final_conf.get(cat)
                conf_str = f" (Confidence: {conf})" if conf is not None else ""
                answers.append(f"{cat}{conf_str}")
            formatted_agent_blocks += ", ".join(answers) + "\n"

    return _render_prompt(question_id, formatted_agent_blocks)

PROMPT_BUILDERS = {
    "scalar": _judge_prompt_scalar,
    "coarse": _judge_prompt_coarse,
    "fine": _judge_prompt_fine,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", required=True, help="List of random seeds for trials")
    parser.add_argument("--questions-path", type=str, required=True, help="Path to questions JSON")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--model-path", type=str, required=True, help="Single model checkpoint path")
    parser.add_argument("--conf-render", choices=["scalar", "coarse", "fine"], required=True,
                        help="How the agents' confidence is rendered into the judge prompt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    construct_question_judge_prompt = PROMPT_BUILDERS[args.conf_render]
    seeds = args.seed

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side="left", padding_side="left")
    model = WhiteboxModel.from_pretrained(args.model_path)

    for seed in seeds:
        set_seed(seed)
        rng = random.Random(seed)

        questions_path = args.questions_path.format(seed=seed)
        output_path = args.output_path.format(seed=seed)

        questions = json.load(open(questions_path))
        current_question = 0
        response_dict = {}

        for q_i in trange(
            current_question,
            len(questions),
            initial=current_question,
            total=len(questions),
        ):
            q_data = questions[q_i]
            question = q_data["question_id"]
            answer = q_data["ground_truth_answer"]

            prompt_orig = construct_question_judge_prompt(q_data)
            ctx_orig = [{"role": "user", "content": prompt_orig}]

            swapped_example = swap_agents_in_example(q_data)
            prompt_swapped = construct_question_judge_prompt(swapped_example)
            ctx_swap = [{"role": "user", "content": prompt_swapped}]

            use_swapped = rng.choice([False, True])
            ctx = ctx_swap if use_swapped else ctx_orig
            prompt_used = prompt_swapped if use_swapped else prompt_orig

            completion = generate_answer_standard(ctx, model, tokenizer, max_new_tokens=1500, num_samples=1)
            response_dict[question] = (completion, prompt_used, answer)

            json.dump(
                response_dict,
                open(output_path, "w", encoding="utf-8"),
                indent=2,
                ensure_ascii=False,
            )
