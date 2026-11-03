"""Post-hoc self-verbalised confidence for the life_events output format.

The agent is asked to score the answer it has just given, in a separate turn.

The elicitation context is `agent_context + [elicit_msg]` — a new list, so the
agent's own history is never mutated — and it carries the full debate history.

`raw_exp_confidences_1_10` is returned fully populated over all 8
categories: a category the model did not score falls back to
MISSING_CONFIDENCE. Parseable but out-of-range scores are clipped to [1, 10].

`missing_*` and `elicitation_raw_text` are for the driver to log what
went unscored."""

import gc
import re
from typing import Dict, List, Optional

import difflib
import numpy as np
import torch

from debate.gen_utils import generate_temperature_sampling
from debate.life_events.common import (
    construct_confidence_elicitation_prompt_cg,
    construct_confidence_elicitation_prompt_fg,
)
from debate.life_events.parsing import parse_formatted_response

CATEGORIES: List[str] = [
    "Mental Health",
    "Physical Health",
    "Abuse & Addiction",
    "Relationship & Loss",
    "Career & Education",
    "Financial & Legal & Societal",
    "Lifestyle & Identity & Environment",
    "None",
]

MISSING_CONFIDENCE = 1

_FUZZY_CUTOFF = 0.7

# One "Label: score" pair anywhere in a block. Comma and newline are excluded
# from the label, so one finditer pass handles both the newline-separated
# explanation block and the comma-separated answer block; "&" is allowed, so
# the category names containing it still match.
# Examples: "- Mental Health: 8" / "None : 10." / "Abuse & Addiction: 3"
_SCORE_TOKEN = re.compile(
    r"(?P<label>[^:：,\n]+?)\s*[:：]\s*(?P<score>\d+(?:\.\d+)?)"
)
_ANSWER_BLOCK = re.compile(r"Answer\s+Confidence\s*[:：]?", re.IGNORECASE)
_CG_SCORE = re.compile(r"Confidence\s*[:：]\s*(?P<score>\d+(?:\.\d+)?)", re.IGNORECASE)


def _clip_score(raw: str) -> Optional[int]:
    """Parse one score token into an int in [1, 10], or None if unusable."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return int(round(min(max(v, 1.0), 10.0)))


def _match_category(raw_label: str) -> Optional[str]:
    """Resolve a possibly-misspelled label onto one of the 8 canonical
    names, difflib at cutoff 0.7 (the value `parse_formatted_response`
    uses). None if nothing clears the threshold."""
    # Strip a leading bullet so exact matching hits before the fuzzy pass.
    raw_label = re.sub(r"^[\s\-\*•]+", "", raw_label).strip()
    for cat in CATEGORIES:
        if raw_label.lower() == cat.lower():
            return cat
    close = difflib.get_close_matches(raw_label, CATEGORIES, n=1, cutoff=_FUZZY_CUTOFF)
    return close[0] if close else None


def _parse_score_block(block: str) -> Dict[str, int]:
    """Pull {canonical category name: score} out of a block of `Label: X` lines."""
    found: Dict[str, int] = {}
    for m in _SCORE_TOKEN.finditer(block):
        cat = _match_category(m.group("label"))
        if cat is None:
            continue
        score = _clip_score(m.group("score"))
        if score is None:
            continue
        found[cat] = score
    return found


def parse_verbalized_scores_fg(text: str) -> Dict[str, Dict[str, int]]:
    """Split a fine-grained elicitation reply into its explanation block and
    its answer block, and parse the scores out of each."""
    split = _ANSWER_BLOCK.search(text)
    if split:
        expl_block = text[: split.start()]
        ans_block = text[split.end() :]
    else:
        expl_block = text
        ans_block = ""

    return {
        "explanation": _parse_score_block(expl_block),
        "answer": _parse_score_block(ans_block),
    }


def parse_verbalized_score_cg(text: str) -> Optional[int]:
    """Pull the single overall score out of a coarse-grained elicitation reply."""
    matches = list(_CG_SCORE.finditer(text))
    if not matches:
        return None
    return _clip_score(matches[-1].group("score"))


def verbalized_option_confidence(
    model,
    tokenizer,
    agent_context,
    conf_method: str = "fg_confidence",
) -> dict:
    """Ask an agent to score the answer it just gave, in a separate turn.

    Takes the model, its tokenizer, the agent's context and `conf_method`
    ("fg_confidence" or "whole_confidence"). Returns the conf_dict the
    next round's message constructor reads, plus the raw reply and the list
    of anything left unscored."""
    if agent_context[-1]["role"] != "assistant":
        raise ValueError(
            "verbalized_option_confidence expects a context ending with the "
            f"assistant answer to score; got role={agent_context[-1]['role']!r}"
        )

    completion = agent_context[-1]["content"]
    selected_answers = parse_formatted_response(completion)["answer"]

    if conf_method == "whole_confidence":
        elicit_msg = {
            "role": "user",
            "content": construct_confidence_elicitation_prompt_cg(),
        }
    else:
        elicit_msg = {
            "role": "user",
            "content": construct_confidence_elicitation_prompt_fg(),
        }

    # Concatenation, not append: `agent_context` is left as it was found.
    n_before = len(agent_context)
    elicit_ctx = agent_context + [elicit_msg]

    scores_text = generate_temperature_sampling(
        model, tokenizer, elicit_ctx, num_samples=1
    )[0]

    assert len(agent_context) == n_before, "elicitation mutated agent_context"

    del elicit_ctx
    gc.collect()
    torch.cuda.empty_cache()

    if conf_method == "whole_confidence":
        return _assemble_cg(scores_text, selected_answers)
    return _assemble_fg(scores_text, selected_answers)


def _assemble_fg(scores_text: str, selected_answers: List[str]) -> dict:
    """Build the fine-grained conf_dict, filling every unscored category with
    MISSING_CONFIDENCE."""
    parsed = parse_verbalized_scores_fg(scores_text)
    expl_scores = parsed["explanation"]
    ans_scores = parsed["answer"]

    missing_expl = [c for c in CATEGORIES if c not in expl_scores]
    raw_exp_1_10 = {
        cat: expl_scores.get(cat, MISSING_CONFIDENCE) for cat in CATEGORIES
    }

    # Keyed on what the model selected.
    missing_ans = [c for c in selected_answers if c not in ans_scores]
    filtered_final_1_10 = {
        cat: ans_scores.get(cat, MISSING_CONFIDENCE) for cat in selected_answers
    }

    return {
        "raw_exp_confidences_1_10": raw_exp_1_10,
        "filtered_final_confidence_1_10": filtered_final_1_10,
        "elicitation_raw_text": scores_text,
        "missing_explanation_categories": missing_expl,
        "missing_answer_categories": missing_ans,
    }


def _assemble_cg(scores_text: str, selected_answers: List[str]) -> dict:
    """Build the coarse-grained conf_dict from the single elicited score."""
    score = parse_verbalized_score_cg(scores_text)
    missing = score is None
    if missing:
        score = MISSING_CONFIDENCE

    # cg only ever needs `final_single_conf_1_10` — that is all
    # `construct_message_single_confidence` reads.
    return {
        "final_single_conf_1_10": score,
        "elicitation_raw_text": scores_text,
        "missing_overall_confidence": missing,
    }
