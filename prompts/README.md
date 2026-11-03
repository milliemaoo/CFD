# Few-shot examples

The prompts embed one few-shot block per task, read from this directory:

| file | task |
|---|---|
| `few_shot_life_events.txt` | `life_events` |
| `few_shot_symptoms.txt` | `symptoms` |
| `few_shot_sharenting.txt` | `sharenting` |

**Put your own few-shot examples in these files.** They are not included here:
they quote post text, and the datasets are not publicly released. See the data
section of the top-level README.

`debate.gen_utils.load_few_shot` reads them from this directory. Set
`CFD_PROMPT_DIR` to keep them somewhere else:

```bash
export CFD_PROMPT_DIR=/path/to/your/few-shot
```
