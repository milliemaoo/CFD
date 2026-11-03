"""Post-hoc self-verbalised confidence debate driver for symptoms.

The answer is generated from the shared `construct_question_prompt_fewshot`,
with no confidence instruction in the prompt; `verbalized_option_confidence`
then asks the same agent to score that answer in a separate turn and the
scores are attached to the assistant message afterwards. --conf-method chooses
per-category or single-score, as in the sampling driver.

Reads the questions JSON at --questions-path and writes {post: (agent_contexts, gold)} to --output-path, one file per seed.
The debate stops as soon as every agent gives the same answer set. Elicitation is skipped on the
round that agrees, since no later round reads those scores."""

import json
import os
import torch
import argparse

from debate.gen_utils import (
    Debate,
    construct_assistant_message,
    generate_temperature_sampling,
)
from debate.symptoms.common import (
    construct_question_prompt_fewshot,
    construct_message_option_explanation_confidence,
    construct_message_single_confidence,
)
from debate.symptoms.parsing import parse_formatted_response
from debate.symptoms.posthoc_selfverbalized import (
    MISSING_CONFIDENCE,
    verbalized_option_confidence,
)

from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer
from debate.seeding import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Symptoms post-hoc self-verbalized runner")
    parser.add_argument("--seed", type=int, nargs="+", required=True, help="List of random seeds for trials")
    parser.add_argument("--questions-path", type=str, required=True, help="Path to questions JSON")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--conf-method", choices=["whole_confidence", "fg_confidence"], default="fg_confidence", help="Confidence method")
    parser.add_argument("--model-paths", type=str, nargs="+", required=True,
                        help="Agent model paths or HF ids")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seed
    rounds = args.rounds
    CONF_METHOD = args.conf_method

    model_paths = args.model_paths
    models = [WhiteboxModel.from_pretrained(m) for m in model_paths]
    tokenizers = [AutoTokenizer.from_pretrained(m, truncation_side="left", padding_side="left") for m in model_paths]
    agents = len(models)

    questions = json.load(open(args.questions_path))
    base_output_path, ext = os.path.splitext(args.output_path)

    for trial_index, seed in enumerate(seeds):
        print(f"Running trial {trial_index + 1} with seed {seed}", flush=True)
        current_question = 0
        response_dict = {}

        for q_i in trange(
            current_question,
            len(questions),
            initial=current_question,
            total=len(questions),
        ):
            q_data = questions[q_i]
            question = q_data["post"]
            answer = q_data["labels"]

            # The shared prompt, with no confidence instruction in it.
            prompt = construct_question_prompt_fewshot(question)

            agent_contexts: Debate = [
                [{"role": "user", "content": prompt}] for agent in range(agents)
            ]

            for round_idx in range(rounds):
                torch.cuda.empty_cache()
                round_info = []  # agent indices that generated this round
                confidences = None

                if round_idx > 0:
                    if CONF_METHOD == "whole_confidence":
                        confidences = [
                            agent[-1]["confidence"].get("final_single_conf_1_10", 1)
                            for agent in agent_contexts
                        ]
                    else:
                        confidences = [agent[-1].get("confidence") for agent in agent_contexts]

                for i, agent_context in enumerate(agent_contexts):
                    set_seed(seed)

                    current_model = models[i]
                    current_tokenizer = tokenizers[i]

                    if confidences is not None:
                        other_confs = confidences[:i] + confidences[i + 1:]
                        if CONF_METHOD == "whole_confidence":
                            message = construct_message_single_confidence(
                                other_agents=agent_contexts[:i] + agent_contexts[i + 1:],
                                other_confidences=other_confs,
                                conv_idx=2 * round_idx - 1,
                                self_agent=agent_contexts[i],
                                self_confidence=confidences[i],
                            )
                        else:
                            message = construct_message_option_explanation_confidence(
                                other_agents=agent_contexts[:i] + agent_contexts[i + 1:],
                                other_confidences=other_confs,
                                conv_idx=2 * round_idx - 1,
                                self_agent=agent_contexts[i],
                                self_confidence=confidences[i],
                            )
                        agent_context.append(message)

                    completion = generate_temperature_sampling(
                        current_model, current_tokenizer, agent_context, num_samples=1
                    )[0]
                    assistant_message = construct_assistant_message(completion)
                    agent_context.append(assistant_message)
                    round_info.append(i)

                # Check if agreed
                answer_sets = [
                    set(parse_formatted_response(ctx[-1]["content"])["answer"])
                    for ctx in agent_contexts
                ]
                agreement = (len({frozenset(s) for s in answer_sets}) == 1)

                # Skipped on agreement: the debate stops here, so no later
                # round reads these scores.
                if not agreement:
                    for i in round_info:
                        conf_dict = verbalized_option_confidence(
                            models[i],
                            tokenizers[i],
                            agent_contexts[i],
                            conf_method=CONF_METHOD,
                        )
                        # The scores ride on the assistant message dict; next
                        # round's construct_message_* reads them back out.
                        agent_contexts[i][-1]["confidence"] = conf_dict

                        # Worth logging: if the whole "Answer Confidence:"
                        # section is missing, the agent enters the next round
                        # claiming minimum confidence in all of its answers.
                        missing_expl = conf_dict.get("missing_explanation_categories") or []
                        missing_ans = conf_dict.get("missing_answer_categories") or []
                        missing_cg = conf_dict.get("missing_overall_confidence", False)
                        if missing_expl or missing_ans or missing_cg:
                            parts = []
                            if missing_expl:
                                parts.append(f"explanation {missing_expl}")
                            if missing_ans:
                                parts.append(f"answer {missing_ans}")
                            if missing_cg:
                                parts.append("overall confidence")
                            print(
                                f"[posthoc_selfverbalized] WARNING q={question} agent={i} "
                                f"round={round_idx}: unscored {'; '.join(parts)} "
                                f"-> fell back to {MISSING_CONFIDENCE}; "
                                f"elicitation reply: {conf_dict['elicitation_raw_text']!r}",
                                flush=True,
                            )

                # Save after the confidences are attached, so a crash leaves a
                # complete round on disk.
                response_dict[question] = (agent_contexts, answer)
                trial_output_file = f"{base_output_path}_trial{trial_index+1}_seed{seed}{ext}"
                json.dump(
                    response_dict,
                    open(trial_output_file, "w"),
                )

                if agreement:
                    break
