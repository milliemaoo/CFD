"""Prompt text for the sharenting task.

`construct_question_prompt_fewshot` builds the Cat-CoT question prompt that
every method shares, so a given seed produces the same round-0 answer however
the confidence is later obtained.

`construct_message_single_confidence` and
`construct_message_option_explanation_confidence` build the round-1 debate
message, carrying one confidence for the whole response and one per category
respectively. `construct_confidence_elicitation_prompt_fg` / `_cg` are the
separate-turn prompts used by the post-hoc self-verbalised method."""

from typing import List

from debate.gen_utils import Conversation, Message, load_few_shot
from debate.sharenting.parsing import parse_formatted_response


FORMAT_INST_CRITICAL = "Please strictly follow the output format exactly as shown below. Do not output your confidence scores. Do not include bold text, markdown, or additional explanation.\n\nExplanation:\n1. Personal Data: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n2. Physical, Mental, or Emotional Health: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n3. Intervention Services: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n4. Disruptive Home Life: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n5. Other/None: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n\nAnswer:\nWrite only the category numbers. Do not include category names or extra words. Use commas to separate multiple labels, if any."


START_PREFIX_OPTION_CONF_SELF = "These are solutions and confidence scores (1 to 10, where higher means more confident) provided by you and the other agents for the given problem. Each category includes an explanation with its own confidence score, and each final selected answer has a single overall confidence score: "

START_PREFIX_DEBUNC_CONF_SELF = "These are solutions and confidence scores (1 to 10, where higher means more confident) provided by you and the other agents for the given problem: "


END_PREFIX_OPTION_CONF_SELF = f"""

Based on your and other agents' opinions and confidence levels, can you provide an updated response?

Before you update your answer, carefully think through the following steps for each category:
1. Reflect on Your Original Analysis: Briefly restate your original reasoning, conclusion, and the key evidence supporting it.
2. Critically Evaluate External Opinions: Analyse the explanations provided by the other agents. Identify any strengths, weaknesses, or potential biases in their reasoning. Point out any evidence or details you think are missing or overemphasised.
3. Synthesise and Update: Considering both your original analysis and the external opinions, provide a comprehensive and objective reasoning that explains whether you should adjust your original conclusion or retain it. Ensure that you discuss both sides before arriving at your final decision.

{FORMAT_INST_CRITICAL}"""


def construct_question_prompt_fewshot(post):
    few_shot_examples = load_few_shot("sharenting")

    prompt_template = """You are given a post and must decide whether it contains any of the sharenting behaviours defined below.

Sharenting is when someone shares personal or sensitive information about a specific child (under 18). It also includes posts that might cause harm to the child or to the child's family's reputation. Self-disclosure about oneself, even when under 18, is not considered sharenting.
If the age of the person whose information is being shared is unknown but familial or social roles are mentioned that signal they are a child (e.g., son, daughter, sibling, nephew, niece, grandchild, schoolchild), assume they are children. If they are aged between 18 and 24 (inclusive), sharenting behaviour applies only when they are described in a child role within the family or social context.

Categories:
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

Instructions:
1. Read the post carefully and evaluate whether it matches each defined sharenting behaviour category.
2. For each category, explain your reasoning. If a category applies, support your answer with direct evidence from the post. If it does not apply, explain why there is insufficient or no evidence. Clearly state "yes" or "no" for each category.
3. Finally, list all categories that apply. If more than one applies, separate them with commas.

Below are some examples:
{few_shot_examples}

Post to Analyse:
Post:
"{post}"

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Output Format:
Explanation:
1. Personal Data: [reason]. So the answer is yes (or is no).
2. Physical, Mental, or Emotional Health: [reason]. So the answer is yes (or is no).
3. Intervention Services: [reason]. So the answer is yes (or is no).
4. Disruptive Home Life: [reason]. So the answer is yes (or is no).
5. Other/None: [reason]. So the answer is yes (or is no).

Answer:
Write only the category numbers. Do not include category names or extra words. Use commas to separate multiple labels, if any."""
    prompt = prompt_template.format(post=post, few_shot_examples=few_shot_examples)

    return prompt


def construct_confidence_elicitation_prompt_fg():
    """Separate-turn prompt asking the agent to score the answer it just gave.

    Appended to a context that already ends with that answer. The agent scores
    only what it already wrote and never restates or revises it, so the answer
    stays identical to the one the other estimators were measured on.

    Scores are keyed by category number because two category names contain a
    comma."""
    return """Now rate your own confidence in the response you just gave.

Do not revise your answer. Do not restate your reasoning. Do not add any explanation.

Output integer confidence scores from 1 to 10 for each category in the explanation and the final answer, where 1 means completely uncertain and 10 means completely certain, based on how certain you are about your reasoning and conclusion. The category numbers are: 1 = Personal Data, 2 = Physical, Mental, or Emotional Health, 3 = Intervention Services, 4 = Disruptive Home Life, 5 = Other/None.

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Explanation Confidence:
1: X
2: X
3: X
4: X
5: X

Answer Confidence:
For each category number that appears in your Answer above, in the same order, use the format Category Number: X, and separate multiple category numbers with commas."""


def construct_confidence_elicitation_prompt_cg():
    """Separate-turn prompt asking the agent for one overall score for the answer
    it just gave."""
    return """Now rate your own confidence in the response you just gave.

Do not revise your answer. Do not restate your reasoning. Do not add any explanation.

Output a single overall integer confidence score from 1 to 10 (where 1 means completely uncertain and 10 means completely certain) that reflects your certainty in your entire response.

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Confidence: X"""


def construct_message_single_confidence(
    other_agents: List[Conversation],
    other_confidences,
    conv_idx: int,
    self_confidence,
    self_agent: Conversation = None,
) -> Message:

    prefix_string = START_PREFIX_DEBUNC_CONF_SELF

    # If the current agent's own response is available, render it first
    if self_agent is not None and self_confidence is not None:
        self_response = self_agent[conv_idx]["content"]
        prefix_string += f"\n\nYour original solution:\n\n{self_response}\n\nConfidence: {self_confidence}"

    for agent, confidence in zip(other_agents, other_confidences):
        agent_response = agent[conv_idx]["content"]
        prefix_string += f"\n\nOne agent solution:\n\n{agent_response}\n\nConfidence: {confidence}"

    prefix_string += END_PREFIX_OPTION_CONF_SELF
    return {"role": "user", "content": prefix_string}


def construct_message_option_explanation_confidence(
    other_agents: List[Conversation],
    other_confidences,
    conv_idx: int,
    self_agent: Conversation = None,
    self_confidence: dict = None
) -> Message:

    prefix_string = START_PREFIX_OPTION_CONF_SELF

    # If the current agent's own response is available, render it first
    if self_agent is not None and self_confidence is not None:
        self_response = self_agent[conv_idx]["content"]
        parsed_response = parse_formatted_response(self_response)
        explanation_confidences = self_confidence.get("raw_exp_confidences_1_10", {})
        filtered_final_confidence = self_confidence.get("filtered_final_confidence_1_10", {})

        explanations_with_conf = {}
        for category, explanation in parsed_response["explanations"].items():
            conf = explanation_confidences.get(category, None) if isinstance(explanation_confidences, dict) else None
            if conf is not None:
                explanations_with_conf[category] = f"{explanation} (Confidence: {conf})"
            else:
                explanations_with_conf[category] = explanation

        answers_with_conf = []
        for ans in parsed_response["answer"]:
            conf = filtered_final_confidence.get(ans, None) if isinstance(filtered_final_confidence, dict) else None
            if conf is not None:
                answers_with_conf.append(f"{ans} (Confidence: {conf})")
            else:
                answers_with_conf.append(ans)

        parsed_self_response = "Explanation:\n"
        for i, cat in enumerate(
            ["Personal Data",
             "Physical, Mental, or Emotional Health",
             "Intervention Services",
             "Disruptive Home Life",
             "Other/None"], start=1):
            explanation_text = explanations_with_conf.get(cat, "No explanation provided.")
            parsed_self_response += f"{i}. {cat}: {explanation_text}\n"

        parsed_self_response += "\nAnswer:\n" + ", ".join(answers_with_conf)
        prefix_string += f"\n\nYour original solution:\n\n{parsed_self_response}"

    for i, (agent, conf_tuple) in enumerate(zip(other_agents, other_confidences)):
        explanation_confidences = conf_tuple["raw_exp_confidences_1_10"]
        filtered_final_confidence = conf_tuple["filtered_final_confidence_1_10"]
        agent_response = agent[conv_idx]["content"]

        # Parse agent_response with parse_formatted_response
        parsed_response = parse_formatted_response(agent_response)

        # Attach a confidence score to each explanation
        explanations_with_conf = {}
        for category, explanation in parsed_response["explanations"].items():
            # If explanation_confidences is a dict, take this category's score
            conf = explanation_confidences.get(category, None) if isinstance(explanation_confidences, dict) else None
            if conf is not None:
                explanations_with_conf[category] = f"{explanation} (Confidence: {conf})"
            else:
                explanations_with_conf[category] = explanation

        # Attach a confidence score to each selected answer
        answers_with_conf = []
        for ans in parsed_response["answer"]:
            conf = filtered_final_confidence.get(ans, None) if isinstance(filtered_final_confidence, dict) else None
            if conf is not None:
                answers_with_conf.append(f"{ans} (Confidence: {conf})")
            else:
                answers_with_conf.append(ans)

        # Rebuild the parsed agent_response string (Explanation plus Answer)
        parsed_agent_response = "Explanation:\n"
        for j, cat in enumerate(
            ["Personal Data",
             "Physical, Mental, or Emotional Health",
             "Intervention Services",
             "Disruptive Home Life",
             "Other/None"], start=1):
            explanation_text = explanations_with_conf.get(cat, "No explanation provided.")
            parsed_agent_response += f"{j}. {cat}: {explanation_text}\n"

        parsed_agent_response += "\nAnswer:\n" + ", ".join(answers_with_conf)
        prefix_string += f"\n\nOne agent solution:\n\n{parsed_agent_response}"

    prefix_string += END_PREFIX_OPTION_CONF_SELF
    return {"role": "user", "content": prefix_string}

