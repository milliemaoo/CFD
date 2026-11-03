# CFD — Confidence-Aware Fine-Grained Debate

Code for *Automated Data Enrichment using Confidence-Aware Fine-Grained Debate
among Open-Source LLMs for Mental Health and Online Safety*
([arXiv:2512.06227](https://arxiv.org/abs/2512.06227)).

CFD enriches a dataset with multi-label annotations in the way human annotators
collaborate. Open-source LLM agents first annotate independently; if they agree,
the labels are accepted. Otherwise a fine-grained debate phase is triggered, in
which each agent is shown the others' judgements and confidence scores alongside
its own before revising. Confidence is estimated **per category** rather than
once per response, so an agent can be confident about one label and unsure about
another.

## Tasks

Three multi-label annotation tasks over social-media posts:

| task | labels |
|---|---|
| `life_events` | 8 life-event categories |
| `symptoms` | 7 symptom categories |
| `sharenting` | 5 sharenting-behaviour categories |

## Confidence estimators

Each estimator has one entry script per task, and `--conf-method` selects the
granularity.

| script | where the confidence comes from |
|---|---|
| `*_multimodel_entropy.py` | token entropy of the agent's own generation |
| `*_multimodel_sampling.py` | agreement across re-samples, scored by NLI entailment and answer frequency |
| `*_multimodel_posthoc_selfverbalized.py` | the agent is asked to score its own answer in a separate turn |

| `--conf-method` | |
|---|---|
| `fg_confidence` | one score per category (default) |
| `whole_confidence` | one score for the whole response |

Where the agents still disagree after the debate, a judge model reads both
transcripts and produces the final label set. `judge_two_swap.py` renders the
transcript three ways:

| `--conf-render` | the judge sees |
|---|---|
| `fine` | a score for each category and each selected label |
| `coarse` | one score per round, taken from the confidence record |
| `scalar` | one score per round, given directly as a number |

## Install

Python 3.10, and a CUDA GPU. FlashAttention-2 is built against the installed
PyTorch, so install it in this order:

```bash
pip install torch==2.4.1
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
python -c "import nltk; nltk.download('punkt')"
```

`pyproject.toml` pins the environment the paper's results were produced in.

Agent models are loaded in 4-bit (`bitsandbytes`) and bfloat16, with
`device_map="auto"` and FlashAttention-2.

The sampling estimator scores explanation consistency with an NLI model:

```bash
export CFD_NLI_MODEL=tasksource/ModernBERT-large-nli   # or a local copy
export CFD_NLI_DEVICE=0                                # -1 for CPU
```

The prompts embed a few-shot block per task, read from `prompts/`. See
`prompts/README.md`.

## Models used in the paper

| role | model |
|---|---|
| agent | `Qwen/Qwen2.5-32B-Instruct` |
| agent | `mistralai/Mistral-Small-24B-Instruct-2501` |
| judge | `meta-llama/Llama-3.3-70B-Instruct` |
| NLI | `tasksource/ModernBERT-large-nli` |

`--model-paths` accepts local directories or Hugging Face ids.

## Running a debate

```bash
export PYTHONHASHSEED=42

python src/debate/life_events/life_events_multimodel_entropy.py \
    --seed 42 43 44 \
    --rounds 2 \
    --conf-method fg_confidence \
    --questions-path  data/life_events/questions.json \
    --output-path     results/life_events/entropy_fine.json \
    --model-paths     Qwen/Qwen2.5-32B-Instruct mistralai/Mistral-Small-24B-Instruct-2501
```

`--conf-method whole_confidence` runs the coarse-grained version of the same
estimator. `*_multimodel_sampling.py` takes a single `--seed` and is run once
per seed.

Output is `{post: (agent_contexts, gold_labels)}`, one file per seed. The
debate stops early once every agent gives the same answer set.

## Running the judge

```bash
python src/debate/life_events/judge_two_swap.py \
    --seed 42 43 44 \
    --conf-render fine \
    --questions-path  data/life_events/judging_data_seed{seed}.json \
    --output-path     results/life_events/judge_fine_seed{seed}.json \
    --model-path      meta-llama/Llama-3.3-70B-Instruct
```

The judge sees the two agents in a random order, so that whichever agent is
shown first does not sway the verdict.

## Repository layout

```
src/debate/<task>/
    common.py                   prompts and debate messages
    parsing.py                  parses the structured response
    fine_grained_token_entropy.py   per-category token entropy
    fine_grained_sampling.py        NLI consistency over re-samples
    posthoc_verbalized.py           separate-turn confidence elicitation
    <task>_multimodel_*.py          debate entry points
    judge_two_swap.py               judge entry point
src/debate/gen_utils.py         generation helpers and conversation types
src/debate/confidence_scaling.py  uncertainty to 1-10 confidence
src/debate/seeding.py           set_seed
src/models/model.py             model wrapper
src/lm_polygraph/               vendored, locally patched
prompts/                        few-shot examples, see prompts/README.md
```

## Data

The datasets are not included. A request route will be pointed to from this
repository after publication.

`prompts/` holds the few-shot examples the prompts embed; see
`prompts/README.md`.

## Citation

```bibtex
@article{mao2025automated,
  title   = {Automated Data Enrichment using Confidence-Aware Fine-Grained
             Debate among Open-Source LLMs for Mental Health and Online Safety},
  author  = {Mao, Junyu and Hills, Anthony and Tseriotou, Talia and
             Liakata, Maria and Shamir, Aya and Sayda, Dan and
             Atzil-Slonim, Dana and Djohari, Natalie and Ugwudike, Pamela and
             Niranjan, Mahesan and Middleton, Stuart E.},
  journal = {arXiv preprint arXiv:2512.06227},
  year    = {2025}
}
```

## License

MIT. See `LICENSE` and `NOTICE`.

## Acknowledgements

This work builds on [DebUnc](https://github.com/lukeyoffe/debunc) (Yoffe et
al.) — thank you for releasing it.

`src/lm_polygraph` is based on the
[LM-Polygraph](https://github.com/IINemo/lm-polygraph) project, which this code
uses to collect per-token log-probabilities and entropy during generation.
`src/models/model.py` includes code from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
The sampling-based estimator follows
[Tanneru et al. (2024)](https://proceedings.mlr.press/v238/harsha-tanneru24a.html).
