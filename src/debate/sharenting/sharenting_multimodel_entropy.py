"""Entropy-based confidence debate driver for sharenting.

Confidence comes from the token entropy of each agent's own generation.
--conf-method picks the granularity:

- "whole_confidence" — one score per agent per round: `MeanTokenEntropy` over
  the whole generation, cross-agent normalised by `unc_to_confidence`, carried
  into the next round by `construct_message_single_confidence`.
- "fg_confidence" — one score per category per agent: per-category mean token
  entropy from `fine_grained_token_entropy`, normalised by
  `fine_grained_unc_to_confidence_global`, carried by
  `construct_message_option_explanation_confidence` as `(Confidence: X)` on
  each of the 5 sharenting behaviour categories and on each selected answer label.

Everything else is shared, so the two settings differ only in where the
confidence comes from.

Reads the questions JSON at --questions-path and writes {post: (agent_contexts, gold)} to --output-path, one file per seed.
The debate stops as soon as every agent gives the same answer set.
"""

import json
import os
import numpy as np
import torch
import argparse

from debate.gen_utils import (
    Debate,
    construct_assistant_message,
    generate_answer_uncertainty,
)
from debate.confidence_scaling import unc_to_confidence
from debate.sharenting.common import (
    construct_question_prompt_fewshot,
    construct_message_single_confidence,
    construct_message_option_explanation_confidence,
)
from debate.sharenting.parsing import parse_formatted_response
from debate.sharenting.fine_grained_token_entropy import (
    generate_answer_fine_grained_uncertainty,
    fine_grained_unc_to_confidence_global,
)
from lm_polygraph.estimators import MeanTokenEntropy
from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer
from debate.seeding import set_seed

ue_method = MeanTokenEntropy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", required=True,
                        help="List of random seeds for trials")
    parser.add_argument("--questions-path", type=str, required=True,
                        help="Path to questions JSON")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--conf-method", choices=["whole_confidence", "fg_confidence"],
                        default="fg_confidence", help="Confidence granularity")
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
    tokenizers = [
        AutoTokenizer.from_pretrained(m, truncation_side="left", padding_side="left")
        for m in model_paths
    ]
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
            prompt = construct_question_prompt_fewshot(question)
            agent_contexts: Debate = [
                [{"role": "user", "content": prompt}] for _ in range(agents)
            ]

            for round_idx in range(rounds):
                torch.cuda.empty_cache()

                # Confidences for the answers the previous round produced.
                confidences = None
                if round_idx != 0:
                    if CONF_METHOD == "whole_confidence":
                        uncertainties = []
                        for agent in agent_contexts:
                            agent = agent[-1]
                            uncertainties.append(agent["uncertainty"])
                        confidences = unc_to_confidence(np.array(uncertainties))
                    else:
                        per_agent_category_entropies = [
                            agent[-1]["category_entropies"] for agent in agent_contexts
                        ]
                        per_agent_selected_answers = [
                            parse_formatted_response(agent[-1]["content"])["answer"]
                            for agent in agent_contexts
                        ]
                        confidences = fine_grained_unc_to_confidence_global(
                            per_agent_category_entropies=per_agent_category_entropies,
                            per_agent_selected_answers=per_agent_selected_answers,
                        )

                for i, agent_context in enumerate(agent_contexts):
                    set_seed(seed)

                    current_model = models[i]
                    current_tokenizer = tokenizers[i]

                    if confidences is not None:
                        agent_contexts_other = (
                            agent_contexts[:i] + agent_contexts[i + 1:]
                        )
                        if CONF_METHOD == "whole_confidence":
                            other_confidences = np.concatenate(
                                (confidences[:i], confidences[i + 1:])
                            )
                            message = construct_message_single_confidence(
                                other_agents=agent_contexts_other,
                                other_confidences=other_confidences,
                                conv_idx=2 * round_idx - 1,
                                self_agent=agent_contexts[i],
                                self_confidence=confidences[i],
                            )
                        else:
                            other_confidences = (
                                confidences[:i] + confidences[i + 1:]
                            )
                            message = construct_message_option_explanation_confidence(
                                other_agents=agent_contexts_other,
                                other_confidences=other_confidences,
                                conv_idx=2 * round_idx - 1,
                                self_agent=agent_contexts[i],
                                self_confidence=confidences[i],
                            )
                        agent_context.append(message)

                    if CONF_METHOD == "whole_confidence":
                        completion, uncertainty = generate_answer_uncertainty(
                            agent_context, current_model, current_tokenizer, ue_method
                        )
                        assistant_message = construct_assistant_message(completion)
                        assistant_message["uncertainty"] = uncertainty
                        agent_context.append(assistant_message)
                    else:
                        (
                            completion,
                            overall_unc,
                            category_entropies,
                            answer_label_entropies,
                        ) = generate_answer_fine_grained_uncertainty(
                            agent_context, current_model, current_tokenizer
                        )

                        missing_cats = [
                            cat for cat, v in category_entropies.items()
                            if isinstance(v, float) and np.isnan(v)
                        ]
                        if missing_cats:
                            print(
                                f"[fine_grained] WARNING q={q_i} agent={i} "
                                f"round={round_idx}: missing categories "
                                f"{missing_cats}; completion: {completion!r}",
                                flush=True,
                            )

                        assistant_message = construct_assistant_message(completion)
                        assistant_message["overall_uncertainty"] = overall_unc
                        assistant_message["category_entropies"] = category_entropies
                        assistant_message["answer_label_entropies"] = answer_label_entropies
                        agent_context.append(assistant_message)

                response_dict[question] = (agent_contexts, answer)
                trial_output_file = f"{base_output_path}_trial{trial_index+1}_seed{seed}{ext}"
                json.dump(
                    response_dict,
                    open(trial_output_file, "w"),
                )

                answer_sets = []
                for agent in agent_contexts:
                    parsed = parse_formatted_response(agent[-1]["content"])
                    answer_sets.append(set(parsed["answer"]))
                if (
                    len(answer_sets) == agents
                    and len({frozenset(s) for s in answer_sets}) == 1
                ):
                    break
