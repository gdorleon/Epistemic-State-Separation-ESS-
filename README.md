# Epistemic State Separation (ESS)

ESS augments a transformer LLM with a lightweight
read-only **Epistemic State Module (ESM)**, trains it with an **Epistemic
Contrastive Objective (ECO)**, and gates generation via the **Epistemic Gating
Mechanism (EGM)**.

## Installation

```bash
git clone <repo-url> ess-hallucination
cd ess-hallucination
pip install -r requirements.txt
pip install -e .
```

You will need a Hugging Face account with access to `meta-llama/Llama-2-7b-chat-hf`
(or substitute any decoder-only causal LM). Set `HF_TOKEN` in your environment.

## Quickstart

```bash
# 1. Build the ECO training corpus from Wikidata
python scripts/prepare_data.py \
    --output data/eco_corpus.jsonl \
    --n_triples 50000 \
    --n_negatives_per_pos 8

# 2. Mine self-generated negatives (uses base model)
python scripts/prepare_data.py \
    --mode self_negatives \
    --model meta-llama/Llama-2-7b-chat-hf \
    --output data/self_negatives.jsonl \
    --n_prompts 5000

# 3. Run three-phase training
python scripts/train.py --config configs/llama2_7b.yaml

# 4. Evaluate on benchmarks
python scripts/evaluate.py \
    --checkpoint outputs/ess_phase3/ \
    --benchmark all
```

## Repository structure

- `ess/model/`   model wrapper, ESM, EGM
- `ess/data/`    Wikidata pipeline, negative mining, ECO dataset
- `ess/training/`  losses and three-phase trainer
- `ess/inference/`  ESS-gated generation
- `ess/evaluation/`  TruthfulQA, FActScore, SelfCheckGPT, calibration, stress tests

## Citation

```bibtex
@inproceedings{ess2026,
  title = {Epistemic State Separation for Hallucination-Aware Language Generation},
  author = {Anonymous},
  year = {2026}
}
```
