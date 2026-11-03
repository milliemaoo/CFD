"""LLM-as-a-judge over symptoms debates.

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
  re-parsed and rendered with `(Confidence: X)` after each of the 7 symptom
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
from debate.symptoms.parsing import parse_formatted_response
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

_FEW_SHOT_EXAMPLES = load_few_shot("symptoms")


def _render_prompt(question_id, formatted_agent_blocks):
    """Wrap one rendered transcript in the shared judge instruction."""
    few_shot_examples = _FEW_SHOT_EXAMPLES
    prompt_template = f"""You are a judge tasked with evaluating a debate between two agents about whether a given post reflects any current symptoms of psychopathology experienced or expressed by the poster themselves, according to the symptom categories defined below. Note that symptoms may co-occur.

For each category, analyze and compare the agents' arguments for accuracy and relevance. Judge whether their reasoning is well supported by the post content and whether it aligns with the given category definitions.

When comparing agents, treat confidence scores as secondary signals. Base your decision primarily on the strength and clarity of evidence in the post. High confidence without explicit textual support should be treated as weak reasoning.

Then, based on this comparison, decide whether the category applies and explain your reasoning.

---

Symptom Categories:
- Fear and Distress:
Includes emotional problems such as feeling sad, loneliness, hopeless, or worried. This covers depressive symptoms (e.g., sadness, lack of interest in activities, feelings of worthlessness or guilt, and changes in appetite or sleep); anxiety symptoms (e.g., excessive worry, fear, nervousness); stress reactions (e.g., feeling overwhelmed or burnt out, emotional exhaustion); and reactions to traumatic events (e.g., flashbacks, intrusive memories, being easily startled).
It also includes obsessions and compulsions (OCD-related); avoidance of feared situations or trauma reminders; emotional numbing or feeling detached from one's emotions; dissociative experiences (feeling disconnected from oneself or surroundings); and irritability or anger outbursts related to emotional distress.
- Suicidal Thoughts:
Distinct from general distress, this involves specific thoughts about ending one's life or engagement in self-harm or suicide-related behaviors. It includes both passive suicidal ideation (e.g., "I wish I wouldn't wake up") and active suicidal ideation (e.g., considering or planning methods of suicide), as well as references to self-harm behaviors (e.g., cutting), or suicide attempts.
- Substance Abuse:
Refers to excessive use of alcohol, drugs, or other substances in a way that is harmful to the individual or others (non-substance addictions, namely behavioral addictions, are not included). This includes excessive drinking or using recreational drugs; unsuccessful efforts to cut down use; spending significant time obtaining, using, or recovering from substances; and continued use despite negative consequences.
- Antisocial or Antagonistic Externalizing Behavior:
Antisocial behavior includes actions that violate societal norms or infringe upon the rights of others, such as breaking rules; physical aggression; blame externalization (e.g., blaming others for one's failures); disregard for societal norms; deceitfulness or theft; failure to meet obligations; engaging in illegal activities; property destruction; bullying. These behaviors involve little or no remorse. It also includes disinhibited behaviors and symptoms such as impulsivity, distractibility, and risk-taking (ADHD-related).
Antagonistic externalizing behavior includes hostile, threatening, or manipulative interactions; extroverted flirts; deliberately provoking or annoying others; attention seeking; rudeness; using others for personal (secondary) gain; and other deliberately over-extroverted behaviors (physical or emotional). A feeling of regret typically follows these behaviors.
- Detachment:
Characterized by withdrawal from social and emotional experiences with others. The withdrawal may stem from a denial of needing others or from despair. This includes avoiding social interactions; limited emotional expression toward others; lack of close relationships; and reduced ability to experience pleasure in social interactions and relationships (i.e., social anhedonia); isolation.
- Others:
Apply this category only when the post describes symptoms that fall into at least one of the following categories: somatoform-related, eating pathology-related, sexual-related, or thought disorder-related symptoms.
Somatoform-related symptoms involve physical symptoms that cause significant distress but may not have clear medical explanations (e.g., unexplained pain such as headaches or muscle aches, unexplained physical sensations, physical fatigue, physical complaints affecting different body systems). These somatic complaints can also stem from anxiety related to health and the body (illness anxiety disorder).
Eating pathology-related symptoms include problematic cognitions and behaviors related to food, eating patterns, body image, and weight (e.g., restrictive eating, binge eating, compensatory behaviors to prevent weight gain, using food as emotional coping, negative fixation on weight or shape).
Sexual-related symptoms involve difficulties with sexual functioning or satisfaction, or sexual addiction that impairs various other functions (e.g., problems with sexual desire, arousal, or performance; pain during sexual activity; distress related to sexual experiences; pornography addiction or excessive masturbation).
Thought disorder-related symptoms relate to disruptions in perception, beliefs, or thought organization (e.g., unusual sensory experiences such as hearing voices, fixed false beliefs, disorganized thinking or speech, or difficulty distinguishing between what's real and what's not).
- None:
Applies when the post does not clearly reflect any of the symptom categories defined above.

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
- Fear and Distress: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Suicidal Thoughts: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Substance Abuse: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Antisocial or Antagonistic Externalizing Behavior: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Detachment: [Compare, explain, and conclude]. So the answer is yes (or is no).
- Others: [Compare, explain, and conclude]. So the answer is yes (or is no).
- None: [Compare, explain, and conclude]. So the answer is yes (or is no).

Answer:
Output exact category names: Fear and Distress, Suicidal Thoughts, Substance Abuse, Antisocial or Antagonistic Externalizing Behavior, Detachment, Others, or None. Use commas to separate multiple labels, if any."""
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

    all_labels = ["Fear and Distress",
            "Suicidal Thoughts",
            "Substance Abuse",
            "Antisocial or Antagonistic Externalizing Behavior",
            "Detachment",
            "Others",
            "None"]

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
    parser.add_argument("--questions-path", type=str, required=True, help="Path template to questions JSON (use {seed})")
    parser.add_argument("--output-path", type=str, required=True, help="Output template JSON file path (use {seed})")
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
