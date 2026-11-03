"""Per-category mean token entropy for the symptoms output format.

`MeanTokenEntropy` averages per-token entropy over a whole generation, giving
one scalar per agent. This module splits the generation into its 7
explanation segments and averages within each, so every category gets its own
uncertainty value.

Category headers are `- <Category Name>:`. The captured label is
resolved against the canonical names with the same difflib cutoff
`parse_formatted_response` uses (0.8), so a mis-spelled category still
lands in the right segment.

Entropy and token ids come from lm_polygraph and cover the generated text
only, never the prompt, so the few-shot examples cannot reach the segment
boundaries or the entropies.

`fine_grained_unc_to_confidence_global` turns the per-agent entropy dicts into
1-10 confidences, pooling all 7N (agent x category) entropies into one
normalisation so that an agent's own categories rank against each other. The
entropy of the "Answer:" line is returned as `answer_label_entropies` for
inspection and is not consumed."""

import re
from typing import Dict, List, Optional, Tuple

import difflib
import numpy as np
from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.manager import UEManager

from debate.confidence_scaling import unc_to_confidence

CATEGORIES: List[str] = [
    "Fear and Distress",
    "Suicidal Thoughts",
    "Substance Abuse",
    "Antisocial or Antagonistic Externalizing Behavior",
    "Detachment",
    "Others",
    "None",
]
_CATEGORIES_SET = set(CATEGORIES)
_FUZZY_CUTOFF = 0.8  # matches `parse_formatted_response`'s difflib cutoff
                     # (stricter than life_events' 0.7)


def _resolve_category(raw_label: str) -> Optional[str]:
    """Map a raw label onto a canonical CATEGORIES entry, exact match first
    then difflib at cutoff 0.8. None if nothing clears the threshold."""
    label = raw_label.strip()
    if label in _CATEGORIES_SET:
        return label
    close = difflib.get_close_matches(label, CATEGORIES, n=1, cutoff=_FUZZY_CUTOFF)
    return close[0] if close else None


class _PerTokenEntropy(Estimator):
    """Pass-through estimator so `man.estimations` carries the raw per-token
    entropy. `level="sequence"` is required: on any other level
    `_flatten_results` would flatten the sample dimension away."""

    def __init__(self):
        super().__init__(["entropy"], "sequence")

    def __str__(self):
        return "PerTokenEntropyRaw"

    def __call__(self, stats):
        return stats["entropy"]


def _run_uemanager(model, input_text: str) -> Tuple[List[float], List[int], str]:
    """Return (per_token_entropy, greedy_token_ids, greedy_text) for one input,
    all three aligned to the generated tokens only."""
    estimator = _PerTokenEntropy()
    man = UEManager(
        Dataset([input_text], [""], batch_size=1),
        model,
        [estimator],
        [],  # generation_metrics
        [],  # ue_metrics
        [],  # processors
        ignore_exceptions=False,
        verbose=False,
    )
    man()

    per_sample = man.estimations.get(("sequence", "PerTokenEntropyRaw"), [])
    if len(per_sample) == 0:
        raise RuntimeError(
            "PerTokenEntropy estimator returned nothing — "
            "EntropyCalculator likely failed. Check the model is a "
            "WhiteboxModel and greedy_log_probs were computed."
        )
    if len(per_sample) > 1:
        raise RuntimeError(
            f"_run_uemanager got {len(per_sample)} samples back; this helper "
            "expects exactly one input. Refactor to a per-sample loop."
        )

    entropy_per_token = list(per_sample[0])
    greedy_tokens = list(man.stats["greedy_tokens"][0])
    greedy_text = man.stats["greedy_texts"][0]
    return entropy_per_token, greedy_tokens, greedy_text


def _build_token_char_offsets(
    token_ids: List[int], tokenizer
) -> Tuple[str, List[Tuple[int, int]]]:
    """Decode prefixes of `token_ids` one token at a time and record each
    token's char span in the result. Incremental decoding avoids the
    SentencePiece leading-space loss that per-token decoding causes."""
    decoded_so_far = ""
    offsets: List[Tuple[int, int]] = []
    for i in range(len(token_ids)):
        decoded_now = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        start, end = len(decoded_so_far), len(decoded_now)
        offsets.append((start, end))
        decoded_so_far = decoded_now
    return decoded_so_far, offsets


_CATEGORY_HEADER_RE = re.compile(
    # Permissive on the label; `_resolve_category` rejects what this
    # over-matches. Anchored at line start so a category named mid-prose
    # cannot open a section.
    r"^\s*(?:[-\*]\s*)?(?P<label>[^:：\n]+?)\s*[:：]",
    flags=re.MULTILINE,
)
_ANSWER_HEADER_RE = re.compile(r"^\s*Answer\s*[:：]", flags=re.IGNORECASE | re.MULTILINE)


def _find_category_char_spans(text: str) -> Dict[str, Tuple[int, int]]:
    """Return {canonical_category: (start_char, end_char)} for every category
    found. A category ends where the next one starts, the last one at
    "Answer:" or end of text. First occurrence of each category wins."""
    headers: List[Tuple[int, str]] = []  # (start_char, canonical_cat_name)
    for m in _CATEGORY_HEADER_RE.finditer(text):
        cat = _resolve_category(m.group("label"))
        if cat is not None:
            headers.append((m.start(), cat))

    if not headers:
        return {}

    seen = set()
    deduped: List[Tuple[int, str]] = []
    for start, cat in headers:
        if cat in seen:
            continue
        deduped.append((start, cat))
        seen.add(cat)
    deduped.sort(key=lambda x: x[0])

    m_ans = _ANSWER_HEADER_RE.search(text)
    answer_start = m_ans.start() if m_ans else len(text)

    spans: Dict[str, Tuple[int, int]] = {}
    for i, (start, cat) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else answer_start
        if end <= start:
            end = answer_start
        spans[cat] = (start, end)
    return spans


def _find_answer_label_char_spans(text: str) -> Dict[str, Tuple[int, int]]:
    """Return {canonical_category: (start_char, end_char)} for the labels on
    the comma-separated "Answer:" line. Diagnostic only."""
    m_ans = _ANSWER_HEADER_RE.search(text)
    if m_ans is None:
        return {}

    # Skip the newline that follows `Answer:` so the `find` below lands on
    # the end of the answer line, not on the header's own newline.
    after_header = text[m_ans.end():]
    leading_after_header = len(after_header) - len(after_header.lstrip())
    content_start = m_ans.end() + leading_after_header
    content = text[content_start:]

    # Everything past the answer line is prose that comma-split cannot
    # resolve.
    eol = content.find("\n")
    answer_text = content[:eol] if eol != -1 else content

    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0  # offset within answer_text
    for piece in answer_text.split(","):
        piece_start = cursor
        cursor += len(piece) + 1  # +1 for the comma we split on

        # Slice off any `(...)` tail before resolving and before measuring.
        resolve_target = piece.split("(", 1)[0] if "(" in piece else piece

        cat = _resolve_category(resolve_target)
        if cat is None or cat in spans:
            continue  # first occurrence per canonical category wins

        leading = len(piece) - len(piece.lstrip())
        label_part = piece[leading:].split("(", 1)[0]
        label_end = len(label_part.rstrip())
        s = content_start + piece_start + leading
        e = s + label_end
        if e > s:
            spans[cat] = (s, e)
    return spans


def _mean_entropy_in_char_span(
    per_token_entropy: List[float],
    token_offsets: List[Tuple[int, int]],
    start_char: int,
    end_char: int,
) -> float:
    """Mean entropy of the tokens overlapping [start_char, end_char)."""
    values: List[float] = []
    n = min(len(token_offsets), len(per_token_entropy))
    for i in range(n):
        s, e = token_offsets[i]
        if e <= start_char or s >= end_char:
            continue
        if s == e:  # zero-width token (e.g. EOS after skip_special_tokens=True)
            continue
        values.append(float(per_token_entropy[i]))
    if not values:
        return float("nan")
    return float(np.mean(values))


def compute_fine_grained_entropies(
    model,
    tokenizer,
    input_text: str,
) -> Dict[str, object]:
    """Generate once and return the overall and per-category mean token
    entropies, the decoded text, the per-token entropies and the token ids.
    A category the regex could not locate gets NaN."""
    per_token_entropy, greedy_tokens, greedy_text = _run_uemanager(model, input_text)

    decoded_text, token_offsets = _build_token_char_offsets(greedy_tokens, tokenizer)
    regex_text = decoded_text if decoded_text else greedy_text

    # Per-category explanation entropies
    cat_spans = _find_category_char_spans(regex_text)
    category_entropies: Dict[str, float] = {}
    for cat in CATEGORIES:
        if cat in cat_spans:
            s, e = cat_spans[cat]
            category_entropies[cat] = _mean_entropy_in_char_span(
                per_token_entropy, token_offsets, s, e
            )
        else:
            category_entropies[cat] = float("nan")

    # Per-answer-label entropies (diagnostic only — not consumed downstream).
    ans_spans = _find_answer_label_char_spans(regex_text)
    answer_label_entropies: Dict[str, float] = {}
    for cat, (s, e) in ans_spans.items():
        answer_label_entropies[cat] = _mean_entropy_in_char_span(
            per_token_entropy, token_offsets, s, e
        )

    overall = (
        float(np.mean(per_token_entropy)) if len(per_token_entropy) > 0 else float("nan")
    )

    return {
        "generation_text": greedy_text,
        "overall_entropy": overall,
        "category_entropies": category_entropies,
        "answer_label_entropies": answer_label_entropies,
        "per_token_entropy": per_token_entropy,
        "greedy_tokens": greedy_tokens,
    }


def generate_answer_fine_grained_uncertainty(
    answer_context,
    model,
    tokenizer,
) -> Tuple[str, float, Dict[str, float], Dict[str, float]]:
    """Generate from `answer_context` and return (text, overall_entropy,
    category_entropies, answer_label_entropies)."""
    prompt = tokenizer.apply_chat_template(
        answer_context, tokenize=False, add_generation_prompt=True
    )
    info = compute_fine_grained_entropies(model, tokenizer, prompt)
    return (
        info["generation_text"],
        info["overall_entropy"],
        info["category_entropies"],
        info["answer_label_entropies"],
    )



def fine_grained_unc_to_confidence_global(
    per_agent_category_entropies: List[Dict[str, float]],
    per_agent_selected_answers: Optional[List[List[str]]] = None,
) -> List[Dict[str, Dict]]:
    """Pool every (agent, category) entropy into one DeBUnc normalisation, so that
    an agent's own categories rank against each other.

    Takes the per-agent {category: entropy} dicts and, optionally, each agent's
    selected answer categories. Returns one dict per agent with
    `raw_exp_confidences_1_10` over all categories and
    `filtered_final_confidence_1_10` restricted to the ones that agent selected.

    Invalid cells (NaN or non-positive) become 1; a matrix with no valid cell
    becomes 5 throughout."""
    n_agents = len(per_agent_category_entropies)
    if n_agents == 0:
        return []
    if per_agent_selected_answers is None:
        per_agent_selected_answers = [[] for _ in range(n_agents)]

    # Every valid cell into one pool, keeping its (i, j) to scatter back.
    flat_entropies: List[float] = []
    positions: List[tuple] = []  # (agent_idx, cat_idx)

    for i, agent_dict in enumerate(per_agent_category_entropies):
        for j, cat in enumerate(CATEGORIES):
            e = agent_dict.get(cat, float("nan"))
            try:
                e_f = float(e)
            except (TypeError, ValueError):
                continue
            if np.isnan(e_f) or e_f <= 0:
                continue
            flat_entropies.append(e_f)
            positions.append((i, j))

    # Invalid cells keep the default of 1.
    matrix = np.ones((n_agents, len(CATEGORIES)), dtype=int)

    if len(flat_entropies) == 0:
        # Nothing parseable anywhere — neutral 5 everywhere.
        matrix.fill(5)
    else:
        confs = unc_to_confidence(np.array(flat_entropies))
        for (i, j), c in zip(positions, confs):
            matrix[i, j] = int(c)

    # filtered_final is keyed by category name here, not by digit as in
    # sharenting's global variant.
    cat_to_idx = {cat: j for j, cat in enumerate(CATEGORIES)}

    results: List[Dict[str, Dict]] = []
    for i in range(n_agents):
        raw_exp = {CATEGORIES[j]: int(matrix[i, j]) for j in range(len(CATEGORIES))}
        filtered_final: Dict[str, int] = {}
        for ans_cat in per_agent_selected_answers[i]:
            j = cat_to_idx.get(ans_cat)
            if j is None:
                continue  # unrecognised answer label — silently drop
            filtered_final[ans_cat] = int(matrix[i, j])
        results.append(
            {
                "raw_exp_confidences_1_10": raw_exp,
                "filtered_final_confidence_1_10": filtered_final,
            }
        )
    return results
