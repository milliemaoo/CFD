"""Sampling-based fine-grained confidence for the sharenting output format.

Scores each category of one generation against `num_samples` further samples
of the same context: NLI bidirectional entailment over the explanation steps,
plus how often each label appears in the sampled answers.

`explanation_option_confidence` is the entry point and returns the same
conf_dict contract as the token-entropy and post-hoc verbalised estimators."""

import os

from collections import Counter

import nltk
import numpy as np
import torch
from transformers import pipeline

from debate.sharenting.parsing import parse_formatted_response

_NLI_MODEL_NAME = os.environ.get("CFD_NLI_MODEL")
if not _NLI_MODEL_NAME:
    raise RuntimeError(
        "Set CFD_NLI_MODEL to tasksource/ModernBERT-large-nli "
        "(a Hugging Face id, or a local copy of it)."
    )

nli_model = pipeline(
    "text-classification",
    model=_NLI_MODEL_NAME,
    device=int(os.environ.get("CFD_NLI_DEVICE", 0)),
    torch_dtype=torch.float16,
    batch_size=16,
)


def get_consistency_score_bool(cot_explanation_1_steps, cot_explanation_2_steps, debug=False):
    """Fraction of explanation steps that find an entailment match in the other
    direction too, plus the per-step match mask. Bidirectional, so a sample
    that drops a step is penalised as much as one that adds a step."""
    text_pairs = []

    for step_1 in cot_explanation_1_steps:
        for step_2 in cot_explanation_2_steps:
            text_pairs.append({"text": step_1['step'], "text_pair": step_2['step']})
    try:
        with torch.no_grad():
            results = nli_model(text_pairs)

    except Exception as e:
        print("Error in NLI inference:")
        print(cot_explanation_1_steps, cot_explanation_2_steps)
        print(text_pairs)
        raise e

    entailment_matrix = np.zeros((len(cot_explanation_1_steps), len(cot_explanation_2_steps)))

    idx = 0
    pair_details = []
    for idx_1, step_1 in enumerate(cot_explanation_1_steps):
        for idx_2, step_2 in enumerate(cot_explanation_2_steps):
            score = results[idx]["score"]
            label = results[idx]["label"]

            # Treat "entailment" as a match
            if label == "entailment" and score >= 0.50:
                entailment_matrix[idx_1, idx_2] = 1
            else:
                entailment_matrix[idx_1, idx_2] = 0
            pair_details.append({
                "pair_index": idx,
                "step_1": step_1["step"],
                "step_2": step_2["step"],
                "score": score,
                "label": label,
            })
            idx += 1

    row_max = entailment_matrix.max(axis=1)  # per original step: does any match exist
    col_max = entailment_matrix.max(axis=0)  # per sampled step

    matched_step_count = sum(col_max) + sum(row_max)
    total_step_count = entailment_matrix.shape[0] + entailment_matrix.shape[1]
    if debug:
        print(f"Matched {matched_step_count} / {total_step_count} Steps!")


    return matched_step_count / total_step_count, row_max

def split_explanation_before_answer(text, trigger="so the answer is"):
    """Split an explanation into sentence-level steps, dropping everything from
    the sentence that states the answer onwards."""
    sentences = nltk.tokenize.sent_tokenize(text)
    filtered_sentences = []
    for sentence in sentences:
        if trigger in sentence.lower():
            break
        filtered_sentences.append(sentence.strip())
    # Nothing split out: fall back to the whole text as a single step
    if not filtered_sentences:
        filtered_sentences = [text.strip()]
    return [{"step": s} for s in filtered_sentences]

def explanation_option_confidence(original_cot_explanation: str, sampling_answers: list, alpha: float = 0.5) -> dict:

    categories = [
        "Personal Data",
        "Physical, Mental, or Emotional Health",
        "Intervention Services",
        "Disruptive Home Life",
        "Other/None",
    ]
    categories_num = ["1", "2", "3", "4", "5"]

    # Parse the original response
    orig_parsed = parse_formatted_response(original_cot_explanation)
    original_answers = orig_parsed["answer"]

    N = len(sampling_answers)

    # Raw explanation confidence
    raw_exp_confidences = {}

    for cat in categories:
        orig_text = orig_parsed["explanations"].get(cat, "")
        if orig_text == "No explanation provided.":
            raw_exp_confidences[cat] = 0.0
            continue

        orig_steps = split_explanation_before_answer(orig_text)
        sim_values = []
        for idx, samp in enumerate(sampling_answers):
            samp_parsed = parse_formatted_response(samp)
            samp_text = samp_parsed["explanations"].get(cat, "")
            samp_steps = split_explanation_before_answer(samp_text)

            # Compare the split steps with get_consistency_score_bool
            score, _ = get_consistency_score_bool(orig_steps, samp_steps)
            sim_values.append(score)

        avg_sim = np.mean(sim_values) if sim_values else 0.0
        raw_exp_confidences[cat] = avg_sim

    # Raw sampling frequency (0-1) and disagreement
    answers_list = []
    frequency_counts = {cat: 0 for cat in categories_num}
    for samp in sampling_answers:
        samp_parsed = parse_formatted_response(samp)
        answer = samp_parsed["answer"]
        # Canonical answer string, used for the disagreement count
        ans_str = ",".join(sorted(answer))
        answers_list.append(ans_str)
        # Count how often each category appears
        for label in answer:
            if label in frequency_counts:
                frequency_counts[label] += 1
    raw_sampling_freq = {cat: (frequency_counts[cat] / N) for cat in categories_num}
    unique_answers = set(answers_list)
    if N > 1:
        disagreement = (len(unique_answers) - 1) / (N - 1)
    else:
        disagreement = 0.0
    sampling_answer_distribution = dict(Counter(answers_list))

    filtered_final_conf = {
        cat: raw_sampling_freq.get(cat, 0.0)
        for cat in original_answers
    }

    final_single_conf = 0.5 * (sum(filtered_final_conf.values()) / len(filtered_final_conf)) \
                  + 0.5 * (sum(raw_exp_confidences.values()) / len(raw_exp_confidences))


    # Map 0-1 onto 1-10
    raw_exp_confidences_1_10 = {k: int(round(v * 9 + 1)) for k, v in raw_exp_confidences.items()}
    filtered_final_confidence_1_10 = {k: int(round(v * 9 + 1)) for k, v in filtered_final_conf.items()}
    final_single_conf_1_10 = int(round(final_single_conf * 9 + 1))

    return {
        "raw_exp_confidences_0_1": raw_exp_confidences,
        "filtered_final_confidence_0_1": filtered_final_conf,
        "all_raw_sampling_freq_0_1": raw_sampling_freq,
        "final_single_conf_0_1": final_single_conf,
        "disagreement": disagreement,
        "sampling_answer_distribution": sampling_answer_distribution,
        "raw_exp_confidences_1_10": raw_exp_confidences_1_10,
        "filtered_final_confidence_1_10": filtered_final_confidence_1_10,
        "final_single_conf_1_10": final_single_conf_1_10
    }

