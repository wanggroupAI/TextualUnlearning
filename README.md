# Textual Unlearning Gives a False Sense of Unlearning

This repository contains the official implementation for the ICML 2025 paper:

**"Textual Unlearning Gives a False Sense of Unlearning"**

## Overview

Machine unlearning has emerged as a crucial technique for removing the influence of specific training data from machine learning models. However, this work demonstrates that existing textual unlearning methods may provide only a **false sense of privacy protection**. We propose three novel attack methods to audit the effectiveness of textual unlearning:

- **U-LiRA+**: An enhanced membership inference attack based on LiRA (Likelihood Ratio Attack) for unleanring auditing
- **TULA-DR**: White-Box Textual Unlearning Attack for Data Reconstruction
- **TULA-MI**: Black-Box Textual Unlearning Attack for Membership Inference


## Repository Structure

```
.
├── U-lira+/                    # U-LiRA+ attack implementation
│   ├── ulira+.py              # Main attack script
│   ├── functions.py           # Core functions for membership inference
│   └── dataset.py             # Dataset loading and processing
├── tula-dr/                    # TULA-DR attack implementation
│   ├── TULA-DR.py             # Main reconstruction attack script
│   ├── args_factory.py        # Argument configuration
│   ├── init_ul.py             # Initialization module
│   └── utilities_ul.py        # Utility functions for unlearning and reconstruction
├── tula-mi/                    # TULA-MI attack implementation
│   ├── TULA-MI.py             # Main membership inference attack script
│   └── functions.py           # Core functions including LightGBM-based attack
├── datasets/                   # Dataset files
│   ├── synthpai_train_age.csv
│   ├── synthpai_test_age.csv
│   ├── synthpai_train_inc.csv
│   └── synthpai_test_inc.csv
└── requirements.txt            # Python dependencies
```

## Requirements

- Python 3.8+
- PyTorch 2.3.0+
- Transformers 4.41.2+
- CUDA-compatible GPU (recommended)

Install dependencies:

```bash
conda create --name tula --file requirements.txt
conda activate tula
```


## Supported Unlearning Methods

Our attacks are evaluated against the following textual unlearning methods:

| Method | Flag | Description |
|--------|------|-------------|
| Gradient Ascent | `ga` | Maximize loss on forget set |
| KL Divergence | `kl` | Maximize KL divergence from original model |
| NPO | `npo` | Negative Preference Optimization |
| Task Vector | `taskVec` | Task vector based unlearning |

## Usage

### 1. U-LiRA+ Attack

U-LiRA+ performs membership inference on unlearned models using an enhanced version of the Likelihood Ratio Attack.


**Arguments:**
- `--dataset`: Dataset name (`synthpai_age` or `synthpai_inc`)
- `--model_name`: Base model (default: `bert-base-uncased`)
- `--unlearn_md`: Unlearning method (`ga`, `kl`, `npo`, `taskVec`)
- `--unlearn_ep`: Number of unlearning epochs
- `--mis_label`: Use mislabeling strategy (0 or 1)
- `--dataset_size`: Training dataset size
- `--batch_size`: Batch size for training

### 2. TULA-DR Attack (Data Reconstruction)

TULA-DR reconstructs the original training data from gradient information exposed during the unlearning process.


**Arguments:**
- `--dataset`: Dataset name
- `--split`: Data split (`val` or `test`)
- `--loss`: Loss function (`cos` or `l2`)
- `--n_inputs`: Number of inputs to reconstruct
- `--n_steps`: Number of optimization steps
- `--init`: Initialization method (`random`, `lm`, `my`, `latent`)
- `--coeff_perplexity`: Perplexity regularization coefficient
- `--use_swaps`: Enable token swapping optimization

### 3. TULA-MI Attack (Membership Inference)

TULA-MI leverages the output difference between the original and unlearned models to perform membership inference.


**Arguments:**
- `--dataset`: Dataset name
- `--model_name`: Model name
- `--unlearn_md`: Unlearning method
- `--mis_label`: Use mislabeling (0 or 1)
- `--score`: Score type (0: hinge, 1: cross-entropy)

## Datasets

We use the SynthPAI dataset with two attribute prediction tasks:

- **synthpai_age**: Age prediction task
- **synthpai_inc**: Income prediction task

Each dataset contains:
- Training set (`*_train_*.csv`)
- Test set (`*_test_*.csv`)

## Evaluation Metrics

Our attacks are evaluated using:

- **AUC**: Area Under the ROC Curve
- **Accuracy**: Attack accuracy
- **TPR@k%FPR**: True Positive Rate at k% False Positive Rate
  - TPR@0.1%FPR
  - TPR@0.5%FPR
  - TPR@1%FPR
  - TPR@5%FPR
- **ROUGE scores** (for TULA-DR): ROUGE-1, ROUGE-2, ROUGE-L

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{du2025textual,
  title={Textual Unlearning Gives a False Sense of Unlearning},
  author={Du, Jiacheng and Wang, Zhibo and Zhang, Jie and Pang, Xiaoyi and Hu, Jiahui and Ren, Kui},
  booktitle={International Conference on Machine Learning},
  pages={14579--14597},
  year={2025},
  organization={PMLR}
}
```

## License

This project is for academic research purposes.

## Acknowledgements

We would like to thank the authors of the following repositories for their excellent work, which served as valuable references for our implementation:

- **[LAMP](https://github.com/eth-sri/lamp)** - Language Model Attack via Paraphrasing by ETH SRI.

- **[LiRA-PyTorch](https://github.com/orientino/lira-pytorch)** - A PyTorch implementation of the Likelihood Ratio Attack (LiRA).

We are grateful for their open-source contributions to the research community.
