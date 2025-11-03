# Feedback Lunch: Deep Feedback Codes for Wiretap Channels

This repository contains the implementation for the paper:  

> **Yingyao Zhou, Natasha Devroye, and Onur Günlü**  
> arXiv:2510.16620 (2025)  
> [[arXiv link]](https://arxiv.org/abs/2510.16620)  

---

## Overview  

This project explores **deep learned feedback codes** designed for **wiretap channels with output feedback**, focusing on the reversely-degraded case.  

The core idea is a **seeded modular design** that combines a **security layer** using hash functions and a **reliability layer** built from a trained encoder–decoder pair. Information leakage is measured following the approach in the paper *["Mutual Information Estimation via f-Divergence and Data Derangements"](https://proceedings.neurips.cc/paper_files/paper/2024/file/bdcfa850adac4a1088153881282ca972-Paper-Conference.pdf)*. Through the use of **feedback**, the legitimate parties achieve **security advantage gain**, even in the **reversely degraded** case.

---

## Code Structure  

```
wiretap-feedback/
├── estimator/          # Mutual information estimator based on f-dime
├── wiretap_gain.py     # Wiretap-lightcode with categorical cross-entropy
├── wiretap_mi.py       # Mutual information estimation
├── wiretap_trade.py    # Wiretap-lightcode with categorical cross-entropy and information leakage included in the loss
```
## Usage


```bash
git clone https://github.com/zyy-cc/wiretap-feedback.git
cd wiretap-feedback
```

Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

### Training and Evaluation

Channel conditions can be modified in the corresponding `parameters_*.py` files.

To train the model, set the training flag to `1`.

To evaluate the model, set the evaluation flag to `0`.

## Citation
If this is helpful, please cite:

```
@article{zhou2025feedback,
  title={Feedback Lunch: Deep Feedback Codes for Wiretap Channels},
  author={Zhou, Yingyao and Devroye, Natasha and G{\"u}nl{\"u}, Onur},
  journal={arXiv preprint arXiv:2510.16620},
  year={2025}
}
```

## Acknowledgment
This repository is highly based on the excellent open-source implementation from:
https://github.com/sravan-ankireddy/lightcode

We sincerely thank the original author for sharing their work, which provided a strong foundation for this project.