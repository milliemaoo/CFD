"""Structured response parser for the sharenting output format.

    Explanation:
    1. Personal Data: ... So the answer is yes (or is no).
    2. Physical, Mental, or Emotional Health: ...
    ... one numbered line per category ...

    Answer:
    3, 4
"""

import re


def parse_formatted_response(response: str) -> dict:
    """Return {"explanations": {category: text}, "answer": [category, ...]}.

    `answer` is a list of digit strings ("1".."5"), in the order they appear
    in the generation, or in CAT_ORDER when inferred from the per-category
    yes/no. Callers must pass it on as-is — sorting or de-duplicating it
    changes the prompt the next round is built from."""

    CAT_ORDER = [
        "Personal Data",
        "Physical, Mental, or Emotional Health",
        "Intervention Services",
        "Disruptive Home Life",
        "Other/None",
    ]

    result = {"explanations": {}, "answer": []}

    printed_response = False
    def warn(msg: str):
        nonlocal printed_response
        print(msg, flush=True)
        if not printed_response:
            print(f"Original response: {response}", flush=True)
            printed_response = True

    # ---------- Extract the Explanation block; fall back to text before Answer ----------
    m_expl = re.search(
        r'Explanation\s*[:：]\s*(?P<block>[\s\S]*?)(?=^\s*Answer\s*[:：]|\Z)',
        response,
        flags=re.IGNORECASE | re.MULTILINE
    )
    if m_expl:
        expl_block = m_expl.group("block")
    else:
        warn("Warning: Explanation block not found; using fallback up to Answer.")
        m_ans_pos = re.search(r'^\s*Answer\s*[:：]', response, flags=re.IGNORECASE | re.MULTILINE)
        expl_block = response[:m_ans_pos.start()] if m_ans_pos else response

    # ---------- Parse explanation lines 1..5 (text kept verbatim) ----------
    line_pattern = re.compile(
        r'^\s*(?P<num>[1-5])\.\s*(?P<label>[^:：]+)\s*[:：]\s*(?P<text>.*)$',
        flags=re.IGNORECASE | re.MULTILINE
    )

    for m in line_pattern.finditer(expl_block):
        num = m.group("num")
        text = m.group("text").strip()
        idx = int(num) - 1
        if 0 <= idx < len(CAT_ORDER):
            result["explanations"][CAT_ORDER[idx]] = text if text else "No explanation provided."

    # ---------- Fill in missing categories and warn ----------
    missing = [lbl for lbl in CAT_ORDER if lbl not in result["explanations"]]
    if missing:
        for lbl in missing:
            result["explanations"][lbl] = "No explanation provided."
        warn(f"Warning: Missing explanations for categories {missing}; set to 'No explanation provided.'")

    # ---------- Prefer the digits on the Answer line ----------
    m_ans = re.search(
        r'^\s*Answer\s*[:：]\s*(?P<ans>.*)$',
        response,
        flags=re.IGNORECASE | re.MULTILINE
    )
    nums_from_answer = []
    if m_ans:
        nums_from_answer = re.findall(r'\b[1-5]\b', m_ans.group("ans"))
        if not nums_from_answer:
            warn("Warning: Answer line found but contains no valid numbers 1-5; will infer from explanations.")

    # ---------- No digits on Answer: infer from the per-category yes/no ----------
    nums_from_expl = []
    if not nums_from_answer:
        yn_pat = re.compile(r'\bSo\s+the\s+answer\s+is\s+(yes|no)\b', flags=re.IGNORECASE)
        for i, label in enumerate(CAT_ORDER, start=1):
            expl_text = result["explanations"].get(label, "")
            m = yn_pat.search(expl_text)
            if m:
                if m.group(1).lower() == "yes":
                    nums_from_expl.append(str(i))
            else:
                pass

        if not nums_from_expl:
            warn("Warning: Could not infer numbers from explanations; will fallback to scan entire response.")

    # ---------- Last resort: scan the whole response for 1..5, dedup, keep order ----------
    if nums_from_answer:
        result["answer"] = nums_from_answer
    elif nums_from_expl:
        result["answer"] = nums_from_expl
    else:
        seen = set()
        nums = []
        for n in re.findall(r'\b[1-5]\b', response):
            if n not in seen:
                nums.append(n)
                seen.add(n)
        if not nums:
            warn("Warning: No numbers 1-5 found anywhere; returning empty answer list.")
        result["answer"] = nums

    return result
