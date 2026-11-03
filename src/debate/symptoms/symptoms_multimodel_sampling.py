"""Sampling-based confidence debate driver for symptoms.

After each round the agent's own context is re-sampled 5 times and
`explanation_option_confidence` scores its categories by NLI consistency
between the explanation steps plus how often each label survives resampling.
--conf-method chooses whether the next round's message carries one score per
category ("fg_confidence") or the single blended score ("whole_confidence").

Reads the questions JSON at --questions-path and writes {post: (agent_contexts, gold)} to --output-path, one file per seed.
The debate stops as soon as every agent gives the same answer set."""

import json
import torch
import argparse

from debate.gen_utils import (
    Debate,
    construct_assistant_message,
)
from debate.symptoms.common import (
    construct_question_prompt_fewshot,
    construct_message_single_confidence,
    construct_message_option_explanation_confidence
)

from debate.gen_utils import generate_temperature_sampling
from debate.symptoms.parsing import parse_formatted_response
from debate.symptoms.fine_grained_sampling import explanation_option_confidence

from models.model import WhiteboxModel
from tqdm import trange
from transformers import AutoTokenizer
from debate.seeding import set_seed

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        device_properties = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {device_properties.name}")
else:
    print("No CUDA-capable devices are available.")

def parse_args():
    parser = argparse.ArgumentParser(description="symptoms Runner")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--questions-path", type=str, required=True, help="Path to questions JSON")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds")
    parser.add_argument("--conf-method", choices=["whole_confidence","fg_confidence"], default="fg_confidence", help="Confidence method")
    parser.add_argument("--model-paths", type=str, nargs="+", required=True,
                        help="Agent model paths or HF ids")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    questions_path = args.questions_path
    output_path = args.output_path
    rounds = args.rounds
    CONF_METHOD = args.conf_method

    model_paths = args.model_paths
    models = [WhiteboxModel.from_pretrained(m) for m in model_paths]
    tokenizers = [AutoTokenizer.from_pretrained(m, truncation_side="left", padding_side="left") for m in model_paths]
    agents = len(models)

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
        question = q_data["post"]
        answer = q_data["labels"]

        # Build the shared question prompt
        prompt = construct_question_prompt_fewshot(question)

        agent_contexts: Debate = [
            [{"role": "user", "content": prompt}] for agent in range(agents)
        ]

        agreement = False
        for round_idx in range(rounds):
            torch.cuda.empty_cache()
            round_info = []  # store (agent_idx, completion, sampling_ctx)
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
                    other_confs = (confidences[:i] + confidences[i + 1:])
                    if CONF_METHOD == "whole_confidence":
                        message = construct_message_single_confidence(
                            other_agents=agent_contexts[:i] + agent_contexts[i + 1:],
                            other_confidences=other_confs,
                            conv_idx=2 * round_idx - 1,
                            self_agent=agent_contexts[i],
                            self_confidence=confidences[i]
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

                # take a snapshot of the context before adding assistant response
                sampling_ctx = agent_context.copy()

                # generate the completion on the snapshot
                completion = generate_temperature_sampling(
                    current_model, current_tokenizer, sampling_ctx, num_samples=1
                )[0]
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                round_info.append((i, completion, sampling_ctx))

            # Check if agreed
            answer_sets = [
                set(parse_formatted_response(ctx[-1]["content"])["answer"])
                for ctx in agent_contexts
            ]
            agreement = (len({frozenset(s) for s in answer_sets}) == 1)

            # Break early on agreement
            if agreement:
                break

            for i, completion, sampling_ctx in round_info:
                model = models[i]
                tokenizer = tokenizers[i]
                sampled_completions = generate_temperature_sampling(
                    model, tokenizer, sampling_ctx, num_samples=5
                )
                conf_dict = explanation_option_confidence(
                    completion, sampled_completions
                )
                # write confidence back to the actual context's last message
                agent_contexts[i][-1]["confidence"] = conf_dict

        # Save this question's results
        response_dict[question] = (agent_contexts, answer)
        json.dump(
            response_dict,
            open(output_path, "w"),
        )
