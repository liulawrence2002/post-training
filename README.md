# Sitefacts

Post-training scaffold for [Thinking Machines Lab Inkling](https://thinkingmachines.ai/inkling/) models on grounded, site-level facts. Training and sampling go through the [Tinker API](https://tinker-docs.thinkingmachines.ai) and [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook). The code in this repo was written against cookbook `0.7.1` and the [Inkling skill](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/inkling/SKILL.md).

The product this is aimed at is a data layer that turns satellite and other physical-world inputs (aerial imagery such as NAIP, elevation, weather, AIS, ADS-B, permits, map layers such as Overture) into LLM-native outputs: time-stamped facts about one site, each with a source and a confidence. The first wedge is property and infrastructure risk (insurers, commercial real estate lenders, utilities). The training target is a model that, given an image tile and/or structured geo context, returns a JSON object of those facts and can call geo tools.

This repository is the training scaffold, not a trained model and not a production data platform. The dataset that ships here is synthetic. It exists so the pipeline, the reward, and the tests run before real labels exist.

## Models

Inkling models on Tinker are already post-trained (instruction-tuned and effort-conditioned). This project continues post-training. It does not do continued pretraining.

| Alias | Tinker id | Params (total / active) | Inputs |
| --- | --- | --- | --- |
| `inkling-2-mini` (default) | `thinkingmachines/Inkling-2-Mini` | 26B / ~5B (pricing page: 5.4B active) | text, images |
| `inkling-2-mini-free` | `thinkingmachines/Inkling-2-Mini:peft:65536:free` | same, 64K, rate-limited, $0 train/sample | text, images |
| `inkling-2-small` | `thinkingmachines/Inkling-2-Small` | 594B / 17B | text, images |
| `inkling` | `thinkingmachines/Inkling` | 975B / 41B | text, images, audio |
| `inkling-small` | `thinkingmachines/Inkling-Small` | 276B / 12B | text, images, audio |

`:peft:262144` ids are the 256K-context training variants. Prices, context, and the free-tier rules change; use [Models & Pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/) rather than this table as the source of truth. Inkling-2-Mini is a looped transformer: on Tinker its middle 12 layers run three times per token, and one LoRA adapter is shared across those passes.

Default config is `thinkingmachines/Inkling-2-Mini` so iteration stays on the inexpensive vision model. Switch with a config file or `--model`:

```bash
python -m sitefacts.train_sft --config configs/inkling-2-small.json
python -m sitefacts.train_sft --model inkling-2-mini-free
```

The cookbook picks the renderer from the model name (`tml_v0` for every current Inkling). This repo calls `model_info.get_recommended_renderer_name` and does not hardcode a renderer. It also does not call `tokenizer.encode` on prompts. The Inkling tokenizer has no chat template, so raw encoding is out of distribution. Do not reuse a tokenizer or a rendered dataset across model names.

## Thinking effort

Every render and every sample passes an explicit effort in `[0.0, 1.0)`. The renderer writes the effort system message. Do not write a second one.

| Preset | `none` | `minimal` | `low` | `medium` | `high` (default) | `xhigh` |
| --- | --- | --- | --- | --- | --- | --- |
| Effort | `0.0` | `0.1` | `0.2` | `0.7` | `0.9` | `0.99` |

The sample set, SFT, RL, and eval all use one effort, default `0.9`, so training tokens and sampling tokens match. Mixed-effort recipes are not implemented. Effort and `max_tokens` are separate. High effort often needs a larger generation budget (16k tokens is a reasonable next step if responses die mid-thought). Sampling temperature stays at `1.0`. Lowering temperature is not a substitute for lowering effort.

## Setup

Requires Python 3.11+. `tinker-cookbook` installs `tinker`, `tml-renderers`, and `torch>=2.10`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: uv venv && uv pip install -e ".[dev]"
```

Create an API key in the [Tinker console](https://tinker-console.thinkingmachines.ai) and export it as `TINKER_API_KEY`. Copy `.env.example` to `.env` if you want the empty variable name locally. The key stays in the environment. Nothing in this repo reads a key out of a committed file, and the training logs print only that the variable is set.

`WANDB_PROJECT` is optional. Leave it unset to skip Weights & Biases.

Unit tests do not need a key and do not import the cookbook:

```bash
pytest
```

## How to run each stage

Run commands from the repo root after `pip install -e .`, or with `PYTHONPATH=src`.

1. **Sample data** (already committed; regenerate if you change the catalog):

   ```bash
   python -m sitefacts.sample_data
   ```

   Optional unlabeled public tiles (NASA GIBS MODIS, no fact labels, gitignored):

   ```bash
   python scripts/fetch_public_samples.py
   ```

2. **Inspect one conversation** without calling Tinker:

   ```bash
   python -m sitefacts.pipeline
   ```

3. **SFT warm-start.** Recommended before RL, because the JSON shape is far from a generic answer.

   ```bash
   python -m sitefacts.train_sft --config configs/default.json
   ```

   The learning rate is `hyperparam_utils.get_lr(model_name)` unless `sft_learning_rate` is set. That value is an AdamW LoRA starting point, not a tuned optimum. In cookbook 0.7.1 the Inkling multipliers differ by a lot (Inkling-Small's multiplier is more than twice Inkling-2-Mini's), so a rate from one model is the wrong rate for another. Move a tuned rate with `hyperparam_utils.transfer_lr`.

4. **RL** on the site-facts reward. Pass the SFT checkpoint when you have one.

   ```bash
   python -m sitefacts.train_rl --config configs/default.json \
     --load-checkpoint tinker://YOUR_SFT_CHECKPOINT
   ```

   The default RL rate is `0.1 * get_lr(model)`. Sweep that and the effort on the first steps before a long run, including higher learning rates. The cookbook logs `optim/entropy`. Groups where every sample gets the same reward contribute no gradient; `remove_constant_reward_groups` is on. Inkling's cookbook defaults (agentic rollout preset, `MinViableGroup`) are left in place.

5. **Eval.**

   ```bash
   python -m sitefacts.evaluate --offline
   python -m sitefacts.evaluate --offline --predictions path/to/preds.jsonl --output runs/eval.json
   python -m sitefacts.evaluate --checkpoint tinker://YOUR_CHECKPOINT --output runs/eval.json
   ```

   Offline with no predictions scores the canonical targets. That checks the scorer. It is not a model score. Online eval samples the test split, records `effort` next to the metrics, and retries a malformed parse once. RL does not retry.

Useful flags: `--model`, `--effort` (`high`, `0.99`, ...), `--data`, `--max-steps`, `--base-url`, `--load-checkpoint`. `eval_every` defaults to `0` so a training launch does not spend extra rollouts until you opt in.

## Data

`data/sample/sites.jsonl` is **synthetic sample data**. Six episodes, four train and two test. Each PNG is a 32×32 solid color. The only fact you can verify by opening the image is `placeholder_swatch_color`. Roof, flood, vegetation, defensible-space, and yard claims repeat analyst notes that are already in the prompt. The port episode hides elevation and vessel count from the prompt and shows a `lookup_site_context` tool trace; the tool reads a dict on the episode, not a live AIS or elevation service. Coordinates are a fake grid (`illustrative_not_surveyed`).

Rows with `label_status` other than `synthetic_sample` are skipped. The public-tile script writes `unlabeled_public` rows for that reason.

The answer object is:

```json
{
  "site_id": "sample-site-roof-001",
  "observed_at": "2024-06-15T16:00:00Z",
  "facts": [
    {
      "claim": "roof_covering",
      "value": "asphalt_shingle",
      "unit": null,
      "confidence": 0.6,
      "source": {
        "kind": "analyst_note",
        "uri": "sample://notes/sample-site-roof-001",
        "observed_at": "2024-06-15T16:00:00Z"
      }
    }
  ]
}
```

Images are passed as cookbook image parts (local path or a `data:` URI of a PNG). Remote URLs are not used; tml-renderers will not fetch them. Download tiles in the data pipeline first.

**TODO(data) / TODO(labels):** replace the catalog with licensed imagery aligned to a site and a time, and with labels a reviewer will stand behind. Confidence values in the sample set are a written convention (0.9 swatch, 0.8 tool, 0.6 notes), not a calibrated probability. **TODO(tools):** `lookup_site_context` should become real elevation, weather, AIS, ADS-B, permit, and map calls, with the same argument shape.

## Reward

`sitefacts.reward` scores a response without the API:

| Piece | Weight | What it measures |
| --- | --- | --- |
| Schema | 0.15 | JSON object matches the site-facts schema |
| Fact F1 | 0.70 | Claim and value against the ground truth (numbers within 1%) |
| Provenance | 0.15 | Matched facts cite the same `kind`, `uri`, and `observed_at` |

Unparsed output scores 0. JSON that fails the schema scores 0.05. A wrong `site_id` keeps the schema term and drops fact and provenance credit. Confidence is required and not scored. A mismatched top-level `observed_at` scales the total by 0.9.

The RL environment gives that score on the final assistant turn. Tool turns score 0 until the answer. A malformed parse ends the episode at 0.

## Layout

```
configs/                  model and run knobs; default is Inkling-2-Mini
src/sitefacts/pipeline.py multimodal SFT conversations from a JSONL file
src/sitefacts/train_sft.py
src/sitefacts/rl_env.py   tool loop + grounded JSON reward
src/sitefacts/train_rl.py
src/sitefacts/evaluate.py online sample or offline score
src/sitefacts/reward.py   schema + fact F1 + provenance
data/sample/              synthetic episodes and swatches
scripts/fetch_public_samples.py
tests/                    pytest, no API key
```

## Open questions

These are unresolved on purpose. The scaffold should not pretend otherwise.

1. What is the claim catalog, the unit system, and the time semantics customers will actually consume? The schema here is a stand-in.
2. Where do supervised labels come from: human review, underwriting files, or weak labels from GIS intersection? The synthetic notes are not a vision signal, except the swatch color.
3. Which imagery products and licenses (NAIP, commercial aerial, customer imagery), at what ground sample distance, and how is a tile aligned to a site and a timestamp?
4. Which live tools and credentials (3DEP, NOAA, AIS, ADS-B, permits, Overture), and what does a tool result look like when a provider is down?
5. Effort and learning rate still need a sweep on real data. `0.9` and `get_lr` / `0.1 * get_lr` are the Inkling guide's starting points.
6. How many images per example (one tile, a time series, several sensors) given image-token cost and the 64K vs 256K training context?
7. How is the eval split built so sites and time do not leak? A random row split of real parcels will overstate accuracy.
8. When is an SFT checkpoint good enough to start RL, and which checkpoint should sampling use?
9. When is the free Mini tier enough, and when is Inkling-2-Small worth the price? Audio models (Inkling, Inkling-Small) are unused here.
10. Should confidence be calibrated, or is it only a schema field? The reward does not score it.
11. What should happen when the model states a claim the tools and the image do not support, beyond the precision term already inside fact F1?

If you use Tinker in research, the cookbook asks for this citation: Thinking Machines Lab, 2026. Tinker. https://thinkingmachines.ai/tinker/.
