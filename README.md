# Unlocking Fine-Grained Translation Quality Estimation in LRMs through Mutually Boosting Implicit and Explicit Reasoning

> EMNLP 2026 (main conference)

This repository contains the **data-processing code and training-launch scripts** for the paper above. All paths are anonymized as `/anonymous/path/...` — update them before running.

## Overview

The experimental pipeline consists of three parts:

1. **Data preprocessing** (`preprocess_data/`): parse raw WMT 2022 / 2023 QE and GeneralMT 2024 evaluation files (sentence-level scores, word-level OK/BAD tags, MQM error marks) into unified JSON records with character-level error spans and scores.
2. **Fine-grained subtask construction** (`subtask_data/`): build the "explicit reasoning" training data of the paper — semantic-unit segmentation of translations, unit/sentence-level error-span detection, and MINOR/MAJOR severity classification — then aggregate the subtask samples into SFT-ready instruction–answer datasets.
3. **Training** (`llama-factory-sft-bash.sh`, `verl-grpo-bash.sh`): LoRA SFT of Qwen3-4B-Thinking with LLaMA-Factory, followed by GRPO RL training with verl using char-level subtask rewards, mutually boosting implicit and explicit quality-estimation reasoning.

## Data Sources

All data comes from the official WMT QE shared-task and MQM human-evaluation repositories:

- [WMT-QE-Task/wmt-qe-2022-data](https://github.com/WMT-QE-Task/wmt-qe-2022-data)
- [WMT-QE-Task/wmt-qe-2023-data](https://github.com/WMT-QE-Task/wmt-qe-2023-data)
- [WMT-QE-Task/wmt-qe-2024-data](https://github.com/WMT-QE-Task/wmt-qe-2024-data)
- [google/wmt-mqm-human-evaluation](https://github.com/google/wmt-mqm-human-evaluation)


## Repository Layout

```text
QE/
├── preprocess_data/                 # raw evaluation files → unified JSON
├── subtask_data/                    # fine-grained subtask data builders
│   ├── semanticUnit/                #   semantic-unit segmentation
│   ├── errorDetection/              #   error-span detection (unit / sequential)
│   ├── errorSeverity/               #   MINOR/MAJOR severity classification
│   ├── allTask/                     #   joint detection + severity in one pass
│   ├── generateLFReasoningData.py   #   LLM-filtered (COT) reasoning data
│   ├── aggregateAndShuffle.py       #   per-language-pair aggregation
│   └── mixed_aggregateAndShuffle.py #   cross-language aggregation
├── llama-factory-sft-bash.sh        # LoRA SFT launcher (LLaMA-Factory)
└── verl-grpo-bash.sh                # GRPO RL launcher (verl + vLLM)
```

Config blocks (language pair, input/output paths, sample sizes) are hardcoded at the top of each script — review them before running.

## Usage

1. Download the datasets listed above, then run the scripts in `preprocess_data/` to obtain the unified JSON records.
2. Run the builders in `subtask_data/`, then the aggregation scripts to produce the SFT dataset registered in LLaMA-Factory's `dataset_info.json`.
3. Launch SFT via `llama-factory-sft-bash.sh`, then GRPO RL via `verl-grpo-bash.sh` (expects `train.parquet`/`test.parquet` under the verl data directory and the custom char-level subtask reward functions under verl's `utils/reward_score/`).

## Citation

```bibtex
@misc{dang2026unlockingfinegrainedtranslationquality,
      title={Unlocking Fine-Grained Translation Quality Estimation in LRMs through Mutually Boosting Implicit and Explicit Reasoning}, 
      author={Renfei Dang and Xinye Wang and Zhejian Lai and Weilu Xu and Shimin Tao and Daimeng Wei and Min Zhang and Shujian Huang},
      year={2026},
      eprint={2605.31378},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.31378}, 
}
```
