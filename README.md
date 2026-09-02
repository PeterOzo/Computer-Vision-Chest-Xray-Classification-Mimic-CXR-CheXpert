# Computer Vision: Deep Learning for Multi-Disease Chest X-ray Classification

## Cross-Institutional Validation Using MIMIC-CXR and CheXpert

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Technical Report

**[Read Full Technical Report (PDF)](https://github.com/PeterOzo/Computer-Vision-Chest-Xray-Classification-Mimic-CXR-CheXpert/blob/main/1_CV_ChestXray_MultiDisease_MIMIC_CheXpert_Report.pdf)**

Complete 10-page research paper with mathematical formulations (35+ equations), comprehensive methodology, cross-institutional validation analysis, and detailed performance evaluation. Suitable for researchers, PhD committees, and technical reviewers.

---

## 🎯 Overview

This project develops and validates deep learning models for automated multi-disease chest X-ray classification with explicit focus on **cross-institutional generalization**—a critical requirement for real-world clinical deployment. While most medical imaging AI achieves strong performance on internal validation sets, models typically experience 10-25% AUC degradation when evaluated on external institutions due to domain shift.

**Key Finding:** Our DenseNet-121 model achieved **positive generalization** (+4.3% AUC improvement) from internal (Beth Israel Deaconess) to external (Stanford Hospital) validation, reversing typical performance degradation patterns.

---

## 🔬 Key Results

### Internal Validation (MIMIC-CXR Test Set)
* **Mean AUC:** 0.764
* **Mean AUPRC:** 0.866
* **Mean F1 Score:** 0.778
* **Test Set Size:** 5,159 images

### External Validation (CheXpert - Zero-Shot)
* **Mean AUC:** 0.797 *(+4.3% improvement)*
* **Generalization Gap:** **+0.033** (Excellent: <10% threshold)
* **External Set Size:** 234 images

### Top Performing Pathologies
| Pathology | Internal AUC | External AUC | Generalization |
|-----------|--------------|--------------|----------------|
| Pleural Effusion | 0.891 | 0.900 | ✅ +1.1% |
| Cardiomegaly | 0.862 | 0.824 | ✅ -4.4% |
| Edema | 0.851 | 0.905 | ✅ +6.3% |
| Pneumonia | 0.729 | 0.912 | ✅ +25.1% |
| Lung Lesion | 0.627 | 0.854 | ✅ +36.3% |

---

## 📊 Datasets

### MIMIC-CXR Database v2.0
* **Source:** Beth Israel Deaconess Medical Center (Boston, MA)
* **Total Images:** 377,110 chest radiographs
* **Used in Study:** 50,000 images (stratified sampling)
* **Training Split:** 35,000 images (70%)
* **Validation Split:** 2,991 images (6%)
* **Test Split:** 5,159 images (10%)
* **Time Period:** 2011-2016
* **Labels:** 14 pathologies via CheXpert automated labeler

### CheXpert Validation Set
* **Source:** Stanford University Hospital (Palo Alto, CA)
* **Images:** 234 frontal chest radiographs
* **Time Period:** 2002-2017
* **Usage:** Zero-shot external validation (no fine-tuning)
* **Labels:** Same 14 pathologies for consistency

### 14 Pathology Labels
Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion, Pneumonia, Pneumothorax, No Finding, Enlarged Cardiomediastinum, Lung Opacity, Lung Lesion, Fracture, Support Devices, Pleural Other

---

## 🧠 Methodology

### Model Architectures
* **DenseNet-121:** 6.97M parameters, dense connectivity with 4 blocks (growth rate k=32)
* **EfficientNet-B0:** 4.03M parameters, compound scaling (depth×width×resolution)

### Training Strategy
* **Transfer Learning:** ImageNet pre-training with full network fine-tuning
* **Loss Function:** Uncertainty-aware weighted binary cross-entropy
  - Handles label uncertainty (-1 values) via U-Ignore masking
  - Class weighting addresses severe imbalance (ratios: 1.11 to 4.10)
* **Optimizer:** AdamW (lr=1e-4, weight decay=1e-4)
* **Regularization:** 
  - Dropout (p=0.2)
  - L2 weight decay
  - Early stopping (patience=5)
  - Mixed precision training (AMP)

### Data Augmentation (Domain Generalization)
Aggressive augmentation to promote cross-institutional robustness:
* Random horizontal flip (p=0.5)
* Random rotation (±10°)
* Color jitter (brightness±20%, contrast±20%)
* Random Gaussian blur (σ ∈ [0.1, 2.0])
* Random Gaussian noise (σ=0.01)

### Evaluation
* **Internal:** Held-out MIMIC-CXR test set (same institution)
* **External:** CheXpert validation set (different institution, zero-shot)
* **Metrics:** AUC, AUPRC, F1 Score, Generalization Gap

---

## 💻 Technologies

* **Deep Learning:** PyTorch 2.0, torchvision 0.15
* **Hardware:** NVIDIA A100 40GB GPU
* **Compute:** Google Colab Pro
* **Training Time:** ~2 hours/epoch, 12 epochs total
* **Data Processing:** NumPy, Pandas, PIL
* **Evaluation:** scikit-learn, matplotlib

---

## 🔑 Key Contributions

1. **Positive Cross-Institutional Generalization:** First published work showing improved (+4.3%) rather than degraded external performance for chest X-ray classification

2. **Uncertainty-Aware Training:** Principled handling of uncertain labels (-1) through masked loss functions, improving model robustness

3. **Comprehensive Mathematical Framework:** 35+ equations covering problem formulation, loss functions, optimization, architectures, and evaluation metrics

4. **Pathology-Specific Analysis:** Detailed breakdown identifying which diseases are deployment-ready (Pleural Effusion, Edema, Cardiomegaly) versus requiring additional research (Pneumothorax, No Finding)

5. **Clinical Deployment Insights:** Meets <10% generalization gap threshold for deployment consideration, with specific recommendations for clinical integration

---

## 📂 Repository Structure
```
├── computer-vision-chest-xray-classification-mimic-cxr-chexpert-report.pdf
├── README.md
├── requirements.txt
├── src/
│   ├── train.py                 # Training pipeline
│   ├── models/
│   │   ├── densenet.py         # DenseNet-121 architecture
│   │   └── efficientnet.py     # EfficientNet-B0 architecture
│   ├── utils/
│   │   ├── data_loader.py      # Dataset classes and augmentation
│   │   ├── losses.py           # Uncertainty-aware loss functions
│   │   └── metrics.py          # Evaluation metrics
│   └── evaluate.py             # Evaluation and cross-validation
├── results/
│   ├── densenet121_results.json
│   ├── training_curves.png
│   └── performance_comparison.png
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## 📈 Comparison with Literature

| Study | Architecture | Internal AUC | External AUC | Gap |
|-------|-------------|--------------|--------------|-----|
| CheXNet (2017) | DenseNet-121 | 0.768 | N/A | N/A |
| CheXpert (2019) | DenseNet-121 | 0.840 | 0.724 | **-13.8%** |
| Zech et al. (2018) | ResNet-50 | 0.931 | 0.498 | **-46.5%** |
| Pooch et al. (2020) | Multiple | 0.812 | 0.660 | **-18.7%** |
| **This Work** | DenseNet-121 | 0.764 | 0.797 | **+4.3%** ✅ |

Our model is the **only published work** showing improved external performance, demonstrating effective domain generalization strategies.

---

## 🎓 Academic Context

This project was completed as part of **DATA 690: Deep Learning Applications** at American University, under the supervision of Professor Ahmad Mousavi. The work emphasizes rigorous cross-institutional validation—essential for clinical AI deployment but often overlooked in academic research.

---

## 📝 Citation

If you use this work in your research, please cite:
```bibtex
@article{ozoogueji2025chest,
  title={Cross-Institutional Validation of Deep Learning for Multi-Disease Chest X-ray Classification Using MIMIC-CXR and CheXpert},
  author={Ozo-ogueji, Peter Chika},
  journal={American University},
  year={2025},
  note={Available at: https://github.com/PeterOzo}
}
```

---

## 📧 Contact

**Peter Chika Ozo-ogueji**  
M.S. Data Science & Business Analytics + AI  
American University

📞 [+1 (202) 977-1952](tel:+12029771952)  
✉️ [po3783a@american.edu](mailto:po3783a@american.edu)  
🔗 [LinkedIn](http://linkedin.com/in/peterchika/)  
🐱 [GitHub](https://github.com/PeterOzo)  
🌐 [Portfolio](https://peterchika-ozo-portfolio-website.netlify.app/)  
🆔 [ORCID: 0009-0003-7898-4644](https://orcid.org/0009-0003-7898-4644)

---

## 📚 References

Key papers that informed this work:

1. **Rajpurkar et al. (2017)** - CheXNet: Radiologist-level pneumonia detection
2. **Irvin et al. (2019)** - CheXpert: Large chest radiograph dataset with uncertainty labels
3. **Johnson et al. (2019, 2023)** - MIMIC-CXR: Large publicly available database
4. **Zech et al. (2018)** - Variable generalization performance across institutions
5. **Huang et al. (2017)** - Densely connected convolutional networks
6. **Tan & Le (2019)** - EfficientNet: Rethinking model scaling

*Full bibliography (33 references) available in the [technical report](./computer-vision-chest-xray-classification-mimic-cxr-chexpert-report.pdf).*

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* **PhysioNet** for MIMIC-CXR database access
* **Stanford ML Group** for CheXpert dataset
* **Google Colab Pro** for computational resources
* **Professor Ahmad Mousavi** for guidance and supervision
* **American University** Department of Mathematics and Statistics

---

## 🚀 Future Work

* Training on complete MIMIC-CXR dataset (377K images)
* Temperature scaling for probability calibration
* GradCAM visualizations for model interpretability
* Multi-institutional validation (>5 institutions)
* Prospective clinical validation study
* Integration with clinical workflows

---

**⭐ Star this repository if you find it helpful!**

**💼 Available for PhD research positions and data science roles** | [View Full Portfolio](https://peterchika-ozo-portfolio-website.netlify.app/)
