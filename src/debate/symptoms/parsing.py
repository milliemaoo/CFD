"""Structured response parser for the symptoms output format.

    Explanation:
    - Fear and Distress: ... So the answer is yes (or is no).
    - Suicidal Thoughts: ...
    ... one line per category ...

    Answer:
    <comma-separated subset of the 7 category names>

Category names are matched by exact string, then difflib at cutoff 0.8, then
a yes/no fallback, then a direct search of the whole response, because models
occasionally mis-spell a category name."""

import re
import difflib


def parse_formatted_response(response: str) -> dict:
    """Return {"explanations": {category: text}, "answer": [category, ...]}.

    `answer` comes out of a set, so its order is arbitrary. Callers must
    pass it on as-is — sorting or de-duplicating it changes the prompt the
    next round is built from."""
    result = {"explanations": {}, "answer": set()}
    printed_response = False
    def warn(msg: str):
        nonlocal printed_response
        print(msg, flush=True)
        if not printed_response:
            print(f"Original response: {response}", flush=True)
            printed_response = True

    expected_categories = {
        "Fear and Distress",
        "Suicidal Thoughts",
        "Substance Abuse",
        "Antisocial or Antagonistic Externalizing Behavior",
        "Detachment",
        "Others",
        "None",
    }
    cat_pattern = (
        r"(Fear and Distress|Suicidal Thoughts|Substance Abuse|"
        r"Antisocial or Antagonistic Externalizing Behavior|Detachment|Others|None)"
    )
    # Try to extract the Explanation part
    expl_match = re.search(r'Explanation\s*[:：](?P<expl>[\s\S]*?)(?=\nAnswer\s*[:：]|$)', response, re.IGNORECASE)
    pattern_ans = re.compile(r'(?im)^Answer\s*[:：]\s*(?P<ans>.*?)(?=\n|$)')
    ans_match = pattern_ans.search(response)
    answer_pos = ans_match.start() if ans_match else len(response)

    if expl_match:
        expl_text = expl_match.group("expl")
    else:
         # Fallback: search the entire response for lines matching the expected format
        warn("Warning: Standard Explanation part not found; trying fallback.")
        expl_text = response[:answer_pos]

    # Process line by line, combining consecutive lines into a block
    lines = expl_text.strip().splitlines()
    current_cat = None
    current_block = []

    for line in lines:
        stripped = line.strip()
        # 1) exact match
        exact = re.match(rf'^\s*(?:[-\*]\s*)?{cat_pattern}\s*[:：]\s*(.*)', stripped, re.IGNORECASE)
        if exact:
            # save last category content
            if current_cat:
                result["explanations"][current_cat] = " ".join(current_block).strip()
            # switch to new category
            current_cat = exact.group(1).strip()
            current_block = [exact.group(2).strip()]
            continue

        # 2) fuzzy match
        fuzzy = re.match(r'^\s*(?:[-\*]\s*)?(?P<label>[^:]+?)\s*[:：]\s*(?P<text>.*)', stripped)
        if fuzzy:
            raw_label = fuzzy.group("label").strip()
            close = difflib.get_close_matches(raw_label, expected_categories, n=1, cutoff=0.8)
            if close:
                if current_cat:
                    result["explanations"][current_cat] = " ".join(current_block).strip()
                current_cat = close[0]
                current_block = [fuzzy.group("text").strip()]
            continue

        # 3) else, fit in current_cat
        if current_cat:
            current_block.append(stripped)

    # Save the last category's content
    if current_cat:
        result["explanations"][current_cat] = " ".join(current_block).strip()

    # Check for missing categories; if any are missing, set their explanation to "No explanation found." and output a warning
    missing = expected_categories - set(result["explanations"].keys())
    if missing:
        for cat in missing:
            result["explanations"][cat] = "No explanation found."
        warn(f"Warning: Missing explanations for categories {missing}; set to 'No explanation found.'.")

    # Try to extract the Answer part (which should have the "Answer:" prefix)
    if ans_match:
        ans_text = ans_match.group("ans")
        selected = [x.strip() for x in ans_text.split(",") if x.strip()]
        for x in selected:
            if "(" in x and ")" in x:
                x_core = x.split("(", 1)[0].strip()
            else:
                x_core = x

            if x_core in expected_categories:
                result["answer"].add(x_core)
                continue

            warn(f"Warning: Answer part found but answers are not standard. Try to extract...")
            mapped = None
            for cat in expected_categories:
                if cat.lower() in x_core.lower() or x_core.lower() in cat.lower():
                    mapped = cat
                    break

            if mapped:
                result["answer"].add(mapped)
            else:
                warn(f"Warning: Answer contains illegal item '{x}'; dropped.")
    else:
        # If Answer part is not found, try to automatically extract the answer from the Explanation sentences
        warn("Warning: Answer part not found; trying to automatically extract the answer from explanations...")
        auto_answer = set()
        pat_yesno = re.compile(r'So\s+the\s+answer\s+is\s+(.*?)(?:\.|$)', re.IGNORECASE)
        pat_alt = re.compile(r'Answer\s*[:：]\s*(.*)', re.IGNORECASE)
        for cat, expl in result["explanations"].items():
            m = pat_yesno.search(expl) or pat_alt.search(expl)
            if not m:
                # fallback: try find yes/no or category names in whole explanation
                lower_expl = expl.lower()
                if "yes" in lower_expl:
                    auto_answer.add(cat)
                    continue
                elif "no" in lower_expl:
                    continue
                else:
                    found_any = False
                    for c in expected_categories:
                        if c.lower() in lower_expl:
                            auto_answer.add(c)
                            found_any = True
                    if not found_any:
                        warn(f"No fallback answer found in explanation for {cat}.")
                continue

            ans_str = m.group(1).strip()
            parts = [p.strip() for p in ans_str.split(",") if p.strip()]
            for part in parts:
                low = part.lower()
                if low == "yes":
                    auto_answer.add(cat)
                    break
                if low == "no":
                    break
                if part in expected_categories:
                    auto_answer.add(part)
                    continue
                else:
                    warn(f"Unknown auto-extracted item for {cat}: '{part}'")

        if auto_answer:
            result["answer"] = auto_answer
        else:
            # Fallback: if automatic extraction fails, try to directly extract possible answer categories from the original response
            warn("Warning: Auto-extraction from explanation failed; trying direct search in full response...")
            found = set()
            for cat in expected_categories:
                if re.search(r'\b' + re.escape(cat) + r'\b', response):
                    found.add(cat)
            if found:
                result["answer"] = found
            else:
                raise ValueError("Warning: Answer part not found!")

    return {
        "explanations": result["explanations"],
        "answer": list(result["answer"])
    }
