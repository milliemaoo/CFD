"""Generation helpers and the conversation types shared by every entry point.

A `Debate` is one `Conversation` per agent, and a `Conversation` is a list of
chat messages that may also carry the uncertainty or confidence attached to
that turn.

`generate_answer_standard` and `generate_temperature_sampling` sample one or
more completions directly; `generate_answer_uncertainty` goes through
lm_polygraph so an uncertainty estimator runs alongside the generation."""

import os
from functools import lru_cache
from pathlib import Path
import gc
from typing import Any, Dict, List, Tuple, TypedDict, Union

import torch
from lm_polygraph.utils.manager import estimate_uncertainty
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    uncertainty: NotRequired[float]
    confidence: NotRequired[float]


Conversation = List[Message]

Debate = List[Conversation]

Debates = Dict[str, Tuple[Debate, Any]]


@lru_cache(maxsize=None)
def load_few_shot(task: str) -> str:
    """Read the few-shot examples for one task.

    The files are not included in the repository: they quote post text, and the
    datasets are not publicly released. See `prompts/README.md`.
    CFD_PROMPT_DIR overrides the directory.
    """
    root = Path(os.environ.get("CFD_PROMPT_DIR")
                or Path(__file__).resolve().parents[2] / "prompts")
    path = root / f"few_shot_{task}.txt"
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. The few-shot examples are not included in "
            "the repository; see prompts/README.md."
        )
    return path.read_text(encoding="utf-8")


def get_len(
    context: Conversation, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
):
    return (
        len(
            tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True
            )
        )
        - 4
    )


def construct_assistant_message(completion) -> Message:
    return {"role": "assistant", "content": completion}


def generate_answer_standard(answer_context, model, tokenizer, max_new_tokens=1500, num_samples=1) -> Union[str, List[str]]:
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenize(prompt).to(model.device())
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            min_new_tokens=2,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_samples
        )
    sequences = outputs.sequences
    idx = inputs["input_ids"].shape[1]
    results = [
        tokenizer.decode(seq[idx:], skip_special_tokens=True)
        for seq in sequences
    ]
    del inputs
    return results[0] if len(results) == 1 else results


def generate_answer_uncertainty(
    answer_context, model, tokenizer, ue_method
) -> Tuple[str, float]:
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    out = estimate_uncertainty(model, ue_method, input_text=prompt)
    return out.generation_text, out.uncertainty


def generate_temperature_sampling(model, tokenizer, question, num_samples):

    prompt = tokenizer.apply_chat_template(
        question,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = model.tokenize(prompt).to(model.device())
    temperature = 0.7

    model_path_lower = getattr(model, "model_path", "").lower()

    if "mistral" in model_path_lower:
        temperature = 0.3
    elif "llama" in model_path_lower:
        temperature = 0.6

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            top_k=20,
            top_p=0.8,
            temperature=temperature,
            min_new_tokens=2,
            return_dict_in_generate=True,
            num_return_sequences=num_samples
        )
    sequences = out.sequences
    sampled_answers = []

    for i in range(num_samples):
        if model.model_type == "CausalLM":
            idx = inputs["input_ids"].shape[1]
            seq = sequences[i, idx:].cpu()
        else:
            seq = sequences[i, 1:].cpu()

        text_length = len(seq)
        for j in range(len(seq)):
            if seq[j] == tokenizer.eos_token_id:
                text_length = j
                break
        cut_text = tokenizer.decode(seq[:text_length], skip_special_tokens=True)
        sampled_answers.append(cut_text)

    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    return sampled_answers

