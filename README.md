# ICU-XAI FusionNet
### A Multimodal Explainable Deep Learning Framework for Early ICU Deterioration Prediction

**Author:** Logina Mahmoud  
**Domain:** Artificial Intelligence | Deep Learning | Computer Vision & NLP Applications in Healthcare | Explainable AI

---

## Overview

ICU-XAI FusionNet is a research-oriented multimodal deep learning framework designed to predict clinical deterioration in Intensive Care Unit (ICU) patients within a 24-hour prediction horizon.

The proposed framework integrates multiple clinical information sources to capture complementary patterns associated with patient deterioration:

- **Temporal physiological data:** Vital signs and laboratory measurements collected over time.
- **Clinical textual data:** Nursing notes, physician notes, and radiology reports.
- **Static clinical information:** Demographic characteristics and baseline patient information.

The framework combines temporal deep learning, biomedical language representations, multimodal feature fusion, and explainable AI to investigate more informative and interpretable ICU risk prediction.

## Research Objective

The primary objective is to investigate whether multimodal learning can improve early identification of ICU patients at risk of clinical deterioration.

Clinical deterioration is defined as the occurrence of at least one of the following events within the next 24 hours:

- In-hospital mortality
- Mechanical ventilation initiation
- Vasopressor initiation
- Septic shock

**Observation window:** 48 hours  
**Prediction horizon:** 24 hours

## Proposed Model Architecture

The architecture consists of three modality-specific branches followed by a multimodal fusion mechanism.

### 1. Structured Temporal Branch

A Gated Recurrent Unit (GRU) network processes sequential physiological measurements and laboratory results.

Proposed architecture:
- GRU Layer 1: 128 hidden units
- Dropout: 0.3
- GRU Layer 2: 128 hidden units
- Temporal Attention Mechanism
- Temporal embedding: 256 dimensions

### 2. Clinical Text Branch

BioClinicalBERT is used to extract contextual representations from clinical documentation.

Processing pipeline:
- Clinical text preprocessing
- Tokenization
- Maximum sequence length: 512
- Attention masking
- CLS embedding extraction
- Linear projection: 768 → 256

### 3. Static Clinical Feature Branch

Static demographic and clinical variables are processed through a dense neural network.

Architecture:
- Linear transformation
- Batch Normalization
- ReLU activation
- Output embedding: 64 dimensions

### 4. Multimodal Fusion

A Gated Cross-Modal Attention mechanism is designed to integrate the three modality-specific representations.

The resulting shared representation captures complementary information from temporal measurements, clinical narratives, and static patient characteristics.

### 5. Prediction Head

The fused representation is passed through fully connected layers:

- 512 → 256
- 256 → 128
- 128 → 1

The final sigmoid activation produces a probability estimate for clinical deterioration.

## Explainable Artificial Intelligence

Interpretability is an important component of ICU-XAI FusionNet.

SHAP (SHapley Additive exPlanations) is included in the research design to investigate model behavior at both global and individual-patient levels.

Planned interpretability outputs include:

- SHAP Summary Plot
- SHAP Feature Importance Bar Plot
- SHAP Dependence Plot
- SHAP Waterfall Plot
- SHAP Force Plot

The objective is to identify influential clinical features and understand their contributions to individual predictions.

## Training Configuration

The proposed training configuration includes:

| Parameter | Configuration |
|---|---|
| Loss Function | Focal Loss |
| Gamma | 2.0 |
| Alpha | 0.25 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.00001 |
| Scheduler | ReduceLROnPlateau |
| Training Split | 70% |
| Validation Split | 15% |
| Test Split | 15% |

Patient-level data splitting is intended to prevent patient leakage between training, validation, and testing.

## Experimental Evaluation

The evaluation framework includes the following metrics:

**Discrimination**
- AUROC
- AUPRC

**Classification**
- Accuracy
- Precision
- Recall
- Specificity
- Sensitivity
- F1 Score
- Matthews Correlation Coefficient

**Calibration**
- Brier Score
- Expected Calibration Error

**Clinical Performance**
- False Alarm Rate
- Positive Predictive Value
- Negative Predictive Value

### Experimental Visualizations

The initial experimental outputs include:

1. Receiver Operating Characteristic (ROC) Curve
2. Precision–Recall Curve
3. Calibration Curve
4. Confusion Matrix

The initial plots indicate that positive-class detection and model discrimination require further investigation. Numerical performance metrics and the underlying evaluation counts should be reported alongside the figures before drawing clinical conclusions.

## Baseline Models

The proposed comparative evaluation includes:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Structured-only GRU
5. Text-only ClinicalBERT
6. SOFA Score approximation

## Datasets and External Validation

The research design identifies:

- **MIMIC-IV:** Primary development dataset.
- **eICU:** Intended external validation dataset.

Dataset usage, cohort construction, and external validation results must be documented based on the experiments actually performed.

Access to these datasets requires compliance with their respective access and data-use requirements.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Biomedical NLP | BioClinicalBERT |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI |
| Dashboard | Streamlit |
| Cloud Infrastructure | Google Cloud Vertex AI |

## Project Structure

The intended repository organization is:

```text
ICU-XAI-FusionNet/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   ├── structured_encoder.py
│   ├── text_encoder.py
│   ├── static_encoder.py
│   ├── fusion.py
│   └── fusionnet.py
│
├── training/
│   ├── train.py
│   ├── loss.py
│   └── config.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── plots.py
│
├── explainability/
│   ├── shap_analysis.py
│   └── patient_explanation.py
│
├── deployment/
│   ├── Dockerfile
│   └── vertex_ai/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── results/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── calibration_curve.png
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

*This represents the intended project structure. Only include files and modules that are actually present in the published repository.*

## Installation

Clone the repository:

```bash
git clone https://github.com/logyfarg/ICU-XAI-FusionNet.git

cd ICU-XAI-FusionNet
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:


```

## Reproducibility

For reproducible experiments, the repository should document:

- Dataset versions and cohort extraction procedures.
- Data preprocessing and feature definitions.
- Patient-level splitting strategy.
- Model configuration and training parameters.
- Random seeds and software versions.
- Evaluation scripts and metric calculations.

Do not upload restricted patient-level clinical data, credentials, or private dataset extracts to GitHub.

## Research Limitations

The initial experimental outputs suggest that the current model requires further optimization and evaluation, particularly regarding positive-class detection and calibration.

Further investigation is required to establish:
- Reliable positive-class prediction.
- Appropriate classification threshold selection.
- Robust external validation.
- Patient-level explainability.
- Generalization across clinical datasets.

The framework is a research prototype and is not intended for clinical diagnosis or autonomous medical decision-making.

## Future Work

Potential future development includes:

- Improving multimodal representation learning.
- Addressing severe class imbalance.
- Performing systematic hyperparameter optimization.
- Completing SHAP-based global and local interpretation.
- Conducting external validation using eICU.
- Developing a FastAPI inference service.
- Integrating a Streamlit monitoring dashboard.
- Exploring deployment through Google Cloud Vertex AI.

## Author

**Logina Mahmoud**

AI & Machine Learning Researcher  
Mechatronics Engineering

GitHub: [@logyfarg](https://github.com/logyfarg)

---

## Acknowledgment

This project explores the application of multimodal deep learning and explainable artificial intelligence to healthcare risk prediction, with a focus on improving the interpretability and evaluation of early ICU deterioration prediction systems.

**Disclaimer:** This repository is intended for academic and research purposes only. It is not a clinically validated medical device.
