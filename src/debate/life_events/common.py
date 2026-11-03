"""Prompt text for the life_events task.

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
from debate.life_events.parsing import parse_formatted_response

FORMAT_INST_CRITICAL = "Please strictly follow the output format exactly as shown below. Do not output your confidence scores. Do not include bold text, markdown, or additional explanation.\n\nExplanation:\n- Mental Health: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Physical Health: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Abuse & Addiction: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Relationship & Loss: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Career & Education: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Financial & Legal & Societal: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- Lifestyle & Identity & Environment: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n- None: [Include your critical reasoning with reference to both your view and others' input]. So the answer is yes (or is no).\n\nAnswer:\nOutput exact category names: Mental Health, Physical Health, Abuse & Addiction, Relationship & Loss, Career & Education, Financial & Legal & Societal, Lifestyle & Identity & Environment, or None. Use commas to separate multiple labels, if any."


START_PREFIX_OPTION_CONF_SELF = "These are solutions and confidence scores (1 to 10, where higher means more confident) provided by you and the other agents for the given problem. Each category includes an explanation with its own confidence score, and each final selected answer has a single overall confidence score: "

START_PREFIX_DEBUNC_CONF_SELF = "These are solutions and confidence scores (1 to 10, where higher means more confident) provided by you and the other agents for the given problem: "


END_PREFIX_OPTION_CONF_SELF = f"""

Based on your and other agents' opinions and confidence levels, can you provide an updated response?

Before you update your answer, carefully think through the following steps for each category:
1. Reflect on Your Original Analysis: Briefly restate your original reasoning, conclusion, and the key evidence supporting it.
2. Critically Evaluate External Opinions: Analyze the explanations provided by the other agents. Identify any strengths, weaknesses, or potential biases in their reasoning. Point out any evidence or details you think are missing or overemphasized.
3. Synthesize and Update: Considering both your original analysis and the external opinions, provide a comprehensive and objective reasoning that explains whether you should adjust your original conclusion or retain it. Ensure that you discuss both sides before arriving at your final decision. 

{FORMAT_INST_CRITICAL}"""


def construct_question_prompt_fewshot(post):
    few_shot_examples = load_few_shot("life_events")

    prompt_template = """You are provided with a social media post and must identify any personal life events based on the life event categories defined below.

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

Instructions:
1. Read the post carefully and evaluate whether it matches each defined life event category.
2. For each life event category, explain your reasoning. If a category applies, support your answer with direct evidence from the post. If it does not apply, explain why there is insufficient or no evidence. Clearly state "yes" or "no" for each category.
3. Finally, list all broad life event categories that apply. If more than one applies, separate them with commas.

Below are some examples:
{few_shot_examples}

Post to Analyze:
Post:
"{post}"

Please strictly follow the output format exactly as shown below. Do not use bold, markdown, or extra formatting.

Output Format:
Explanation:
- Mental Health: [reason]. So the answer is yes (or is no).
- Physical Health: [reason]. So the answer is yes (or is no).
- Abuse & Addiction: [reason]. So the answer is yes (or is no).
- Relationship & Loss: [reason]. So the answer is yes (or is no).
- Career & Education: [reason]. So the answer is yes (or is no).
- Financial & Legal & Societal: [reason]. So the answer is yes (or is no).
- Lifestyle & Identity & Environment: [reason]. So the answer is yes (or is no).
- None: [reason]. So the answer is yes (or is no).

Answer:
Output exact category names: Mental Health, Physical Health, Abuse & Addiction, Relationship & Loss, Career & Education, Financial & Legal & Societal, Lifestyle & Identity & Environment, or None. Use commas to separate multiple labels, if any."""
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
- Mental Health: X
- Physical Health: X
- Abuse & Addiction: X
- Relationship & Loss: X
- Career & Education: X
- Financial & Legal & Societal: X
- Lifestyle & Identity & Environment: X
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
        for cat in ["Mental Health", "Physical Health", "Abuse & Addiction", "Relationship & Loss", "Career & Education", "Financial & Legal & Societal", "Lifestyle & Identity & Environment", "None"]:
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
        for cat in ["Mental Health", "Physical Health", "Abuse & Addiction", "Relationship & Loss", "Career & Education", "Financial & Legal & Societal", "Lifestyle & Identity & Environment", "None"]:
            explanation_text = explanations_with_conf.get(cat, "No explanation found.")
            parsed_agent_response += f"- {cat}: {explanation_text}\n"

        parsed_agent_response += "\nAnswer:\n" + ", ".join(answers_with_conf)
        prefix_string += f"\n\nOne agent solution:\n\n{parsed_agent_response}"

    prefix_string += END_PREFIX_OPTION_CONF_SELF
    return {"role": "user", "content": prefix_string}

