"""Prompt text for the symptoms task.

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
from debate.symptoms.parsing import parse_formatted_response


FORMAT_INST_CRITICAL = "Please strictly follow the output format exactly as shown below. Do not output your confidence scores. Do not include bold text, markdown, or additional explanation.\n\nExplanation:\n- Fear and Distress: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Suicidal Thoughts: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Substance Abuse: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Antisocial or Antagonistic Externalizing Behavior: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Detachment: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Others: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- None: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n\nAnswer:\nOutput exact category names: Fear and Distress, Suicidal Thoughts, Substance Abuse, Antisocial or Antagonistic Externalizing Behavior, Detachment, Others, or None. Use commas to separate multiple labels, if any."


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
    few_shot_examples = load_few_shot("symptoms")

    prompt_template = """You are provided with a social media post and must identify any current symptoms of psychopathology experienced or expressed by the poster themselves, according to the symptom categories defined below, noting that symptoms may co-occur.

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

Instructions:
1. Read the post carefully and evaluate whether it matches each defined symptom category.
2. For each category, explain your reasoning. If a category applies, support your answer with direct evidence from the post. If it does not apply, explain why there is insufficient or no evidence. Clearly state "yes" or "no" for each category.
3. Finally, list all applicable symptom categories that are present in the post. If more than one applies, separate them with commas.

Below are some examples:
{few_shot_examples}

Post to Analyze:
Post:
"{post}"

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Output Format:
Explanation:
- Fear and Distress: [reason]. So the answer is yes (or is no).
- Suicidal Thoughts: [reason]. So the answer is yes (or is no).
- Substance Abuse: [reason]. So the answer is yes (or is no).
- Antisocial or Antagonistic Externalizing Behavior: [reason]. So the answer is yes (or is no).
- Detachment: [reason]. So the answer is yes (or is no).
- Others: [reason]. So the answer is yes (or is no).
- None: [reason]. So the answer is yes (or is no).

Answer:
Output exact category names: Fear and Distress, Suicidal Thoughts, Substance Abuse, Antisocial or Antagonistic Externalizing Behavior, Detachment, Others, or None. Use commas to separate multiple labels, if any."""
    prompt = prompt_template.format(post=post, few_shot_examples=few_shot_examples)

    return prompt


def construct_confidence_elicitation_prompt_fg():
    """Separate-turn prompt asking the agent to score the answer it just gave,
    one score per category explanation and one per selected answer label.

    Appended to a context that already ends with that answer. The agent scores
    only what it already wrote and never restates or revises it, so the answer
    stays identical to the one the other estimators were measured on."""
    return """Now rate your own confidence in the response you just gave.

Do not revise your answer. Do not restate your reasoning. Do not add any explanation.

Output integer confidence scores from 1 to 10 for each category in the explanation and the final answer, where 1 means completely uncertain and 10 means completely certain, based on how certain you are about your reasoning and conclusion.

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Explanation Confidence:
- Fear and Distress: X
- Suicidal Thoughts: X
- Substance Abuse: X
- Antisocial or Antagonistic Externalizing Behavior: X
- Detachment: X
- Others: X
- None: X

Answer Confidence:
For each category that appears in your Answer above, in the same order, use the format Category Name: X, and separate multiple categories with commas."""


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

    parsed_answers = {}
    # If the current agent's own response is available, render it first
    if self_agent is not None and self_confidence is not None:
        self_response = self_agent[conv_idx]["content"]
        parsed_response = parse_formatted_response(self_response)
        parsed_answers["self"] = set(parsed_response["answer"])
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
            ["Fear and Distress",
            "Suicidal Thoughts",
            "Substance Abuse",
            "Antisocial or Antagonistic Externalizing Behavior",
            "Detachment",
            "Others",
            "None"], start=1):
            explanation_text = explanations_with_conf.get(cat, "No explanation found.")
            parsed_self_response += f"- {cat}: {explanation_text}\n"

        parsed_self_response += "\nAnswer:\n" + ", ".join(answers_with_conf)
        prefix_string += f"\n\nYour original solution:\n\n{parsed_self_response}"

    for i, (agent, conf_tuple) in enumerate(zip(other_agents, other_confidences)):
        explanation_confidences = conf_tuple["raw_exp_confidences_1_10"]
        filtered_final_confidence = conf_tuple["filtered_final_confidence_1_10"]
        agent_response = agent[conv_idx]["content"]

        # Parse agent_response with parse_formatted_response
        parsed_response = parse_formatted_response(agent_response)
        parsed_answers[f"agent_{i}"] = set(parsed_response["answer"])

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
            ["Fear and Distress",
            "Suicidal Thoughts",
            "Substance Abuse",
            "Antisocial or Antagonistic Externalizing Behavior",
            "Detachment",
            "Others",
            "None"], start=1):
            explanation_text = explanations_with_conf.get(cat, "No explanation found.")
            parsed_agent_response += f"- {cat}: {explanation_text}\n"

        parsed_agent_response += "\nAnswer:\n" + ", ".join(answers_with_conf)
        prefix_string += f"\n\nOne agent solution:\n\n{parsed_agent_response}"

    prefix_string += END_PREFIX_OPTION_CONF_SELF
    return {"role": "user", "content": prefix_string}

