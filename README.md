<div align="center">

# Full-Atom Peptide Design via Riemannian–Euclidean Bayesian Flow Networks

[![Conference](https://img.shields.io/badge/AAAI-2026-blue.svg)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)]()

</div>

---

## 🧬 Overview

This repository hosts the **official implementation** of our AAAI 2026 paper:

📄 **Full-Atom Peptide Design via Riemannian-Euclidean Bayesian Flow Networks**  
🔗 [Paper (arXiv)](https://arxiv.org/pdf/2511.14516)
---

## 📦 Data

To facilitate the reproducibility of our results, we provide the **dataset**, **pretrained models**, and **generated peptides** used in the paper.

All resources are available at:

**Zenodo:** https://doi.org/10.5281/zenodo.18857171

---

## ⚙️ Environment

Create the conda environment using:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate peptide
```

---

## 🚀 Training

Run the following command to train the model:

```bash
python train_bfn.py
```

---

## 🧪 Testing

To evaluate a trained checkpoint:

```bash
python train_bfn.py --config CONFIG_PATH --test_only --test_ckpt_path CHECKPOINT_PATH
```

---

## 📊 Evaluation

Run the evaluation script:

```bash
bash train_eval.sh ROOT_DIR
```

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{qian2025full,
  title={Full-Atom Peptide Design via Riemannian-Euclidean Bayesian Flow Networks},
  author={Qian, Hao and Tu, Shikui and Xu, Lei},
  journal={arXiv preprint arXiv:2511.14516},
  year={2025}
}
```