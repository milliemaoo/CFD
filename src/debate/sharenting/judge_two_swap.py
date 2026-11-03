"""LLM-as-a-judge over sharenting debates.

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
  re-parsed and rendered with `(Confidence: X)` after each of the 5 sharenting behaviour
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
from debate.sharenting.parsing import parse_formatted_response
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

_FEW_SHOT_EXAMPLES = load_few_shot("sharenting")


def _render_prompt(question_id, formatted_agent_blocks):
    """Wrap one rendered transcript in the shared judge instruction."""
    few_shot_examples = _FEW_SHOT_EXAMPLES
    prompt_template = f"""You are a judge tasked with evaluating a debate between two agents about whether a given post reflects one or more predefined sharenting behaviour categories.

For each category, analyse and compare the agents' arguments for accuracy and relevance. Judge whether their reasoning is well supported by the post content and whether it aligns with the given category definitions.

When comparing agents, treat confidence scores as secondary signals. Base your decision primarily on the strength and clarity of evidence in the post. High confidence without explicit textual support should be treated as weak reasoning.

Then, based on this comparison, decide whether the category applies and explain your reasoning.

---

Sharenting Behaviour Definition:

Sharenting is when someone shares personal or sensitive information about a specific child (under 18). It also includes posts that might cause harm to the child or to the child's family's reputation. Self-disclosure about oneself, even when under 18, is not considered sharenting.
If the age of the person whose information is being shared is unknown but familial or social roles are mentioned that signal they are a child (e.g., son, daughter, sibling, nephew, niece, grandchild, schoolchild), assume they are children. If they are aged between 18 and 24 (inclusive), sharenting behaviour applies only when they are described in a child role within the family or social context.

1. Personal Data:
This category includes mentions of a child's GDPR-related personal data, such as:
- A child's name (even if partially given), specific age or a specific school year (e.g., Reception, Year 1-11, Sixth Form), and location details such as city, town, or county, but not country. It also includes sensitive attributes such as sexual orientation, race or ethnicity, religious beliefs, or political opinions.
- Sharing of images, videos, or audio recordings of a child.
2. Physical, Mental, or Emotional Health:
This category includes mentions of a child's physical, mental, or emotional issues, such as:
- Mentions of a child's diagnosed or suspected medical conditions and disorders including ADHD, anxiety, autism, dyslexia, dysgraphia, speech delays, self-harm, or eating disorders.
- Seeking official referrals or a diagnosis for a child, or disclosing that a child is attending therapy.
- Childhood ailments such as nappy rash, vomiting, toothache, or other aches.
- Seeking advice on feeding, weaning, sleep, or potty training.
- Discussion of puberty, a child's emotional distress (e.g., ongoing sadness, anxiety, withdrawal), or a child engaged in negative behaviour such as school refusal, loneliness or not having friends, severe sibling rivalry, violence against siblings or parents, low self-esteem, lying, stealing, bullying, or running away.
- Meltdowns or common behaviours in children under five (e.g., tantrums, hitting, biting, screaming).
- A child being emotionally upset, for example showing a few tears, which is not serious or chronic distress.
Note: Emotional distress should only be marked when the post clearly describes symptoms of a child's emotional distress, such as feeling hopeless, anxious, or withdrawn. Do not over-infer distress based on external causes, negative experiences, or assumptions about what the child might feel.
3. Intervention Services:
This category covers formal institutional or statutory child-focused programmes or casework where the child is referred into or managed by a public or specialist service, such as:
- Social services, social workers, or specialist educational needs and health services, including SENCO, CAMHS, and SENAR.
- An EHCP, the SEN register, DLA (Disability Living Allowance), or an IDP (Independent Development Plan).
- Serious intervention or safeguarding measures such as Section 47 of the Children Act 1989, a CIN plan (Children in Need), a Child Protection Plan, and the child protection register.
- Child criminal behaviour, including criminal allegations, police investigations of a child, or court proceedings against a child (e.g., youth court, juvenile offences).
- Child placed in temporary foster care, child 'in care', or child subject to a Special Guardianship Order (SGO).
4. Disruptive Home Life:
This category includes mentions of a damaging or disruptive home life of a child that can negatively affect a child's physical, emotional, or mental well-being, or that may cause distress to the child and harm their welfare if publicly known.
It applies where there is an indication of a parental or carer relationship (e.g., mum, dad, custody case) or a reasonable suggestion that the child is living in the home or has home-like contact with the parties at least some of the time.
- Domestic violence, non-molestation orders, abuse, or neglect.
- Serious disclosures about parents or any adult living with the child (e.g., a sibling aged 25 or over, or a grandparent), including involvement in serious crimes, drug abuse, disclosures of mental health issues, or seriously disparaging a parent's reputation (e.g., accusations of parental alienation behaviours).
- A parent saying they cannot cope with the child, a parent in emotional distress because of the child, or family conflict because of the child (for example, the child's misbehaviour leading to parental disagreement).
- Refusal to pay child benefit, or a parent rejecting a paternity claim.
- Minor breaches of arrangement orders, minor co-parenting conflicts, parental separation or divorce, CAFCASS and family courts, Section 7 of the Children Act 1989, C100 applications, ICFA, child arrangement orders (CAO), or custody of the child.
5. Other/None:
Applies when the post does not clearly reflect any of the above sharenting categories. For example, spam, advertisements, news, general non-sensitive sharenting about a child's life (e.g., deregistering from school, favourite toys, activities they do, or things they like and dislike), or general advice-seeking (for example, asking about products, items, or activities for a child that may suggest the child's preferences or aspects of the child's life, such as nightlights or nappies).

Constraint: If any of categories 1-4 is "yes", then 5 (Other/None) must be "no". If and only if all of 1-4 are "no", then 5 must be "yes" (and alone).

---

Below are some examples:
{few_shot_examples}

Post to Analyse:
"{question_id}"

Transcript of the Debate:
{formatted_agent_blocks}
---

Please follow the exact format below.

Output Format:
Explanation:
1. Personal Data: [Compare, explain, and conclude]. So the answer is yes (or is no).
2. Physical, Mental, or Emotional Health: [Compare, explain, and conclude]. So the answer is yes (or is no).
3. Intervention Services: [Compare, explain, and conclude]. So the answer is yes (or is no).
4. Disruptive Home Life: [Compare, explain, and conclude]. So the answer is yes (or is no).
5. Other/None: [Compare, explain, and conclude]. So the answer is yes (or is no).

Answer:
Write only the category numbers. Do not include category names or extra words. Use commas to separate multiple labels, if any."""
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

    all_labels = ["Personal Data",
             "Physical, Mental, or Emotional Health",
             "Intervention Services",
             "Disruptive Home Life",
             "Other/None"]

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

    all_labels = ["Personal Data",
             "Physical, Mental, or Emotional Health",
             "Intervention Services",
             "Disruptive Home Life",
             "Other/None"]

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
            for i, cat in enumerate(all_labels, start=1):
                explanation = parsed["explanations"].get(cat, "No explanation found.")
                conf = raw_conf.get(cat)
                conf_str = f" (Confidence: {conf})" if conf is not None else ""
                formatted_agent_blocks += f"{i}. {cat}: {explanation}{conf_str}\n"

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
