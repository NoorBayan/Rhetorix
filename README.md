# 📖 Rhetorix: Distance-Aware Ordinal Learning for Processing-Effort Classification in Classical Arabic Rhetoric

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WGqLul_ENM3dytF-7V7bTtYlT36GFNb5?usp=sharing)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC--BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the dataset, source code, and reproducibility pipeline for the AI Application paper focusing on predicting processing effort in Classical Arabic rhetorical texts. It formulates the prediction of expert-derived figurative interpretation effort as a distance-aware ordinal classification task.

---

## 🏛️ For Humanities, Literature, and Linguistics Scholars
Are you a researcher in **Literary Theory, Applied Linguistics, Rhetoric, or Education**? This project is built with your theories in mind. 

We recognize that language is not just zeros and ones. By integrating concepts like **Relevance Theory** and **Cognitive Pragmatics**, this computational framework offers tangible tools for your research:
* 📚 **For Literary Critics & Theorists:** Explore how abstract theories of meaning-making (like graded salience and inferential burden) can be computationally modeled and validated.
* 🖋️ **For Rhetoric Scholars:** Move beyond the simple binary question of "Is this a metaphor?" to a deeper understanding of textual density in Classical Arabic.
* 🎓 **For Educators:** Explore how these processing-effort models can inform future text readability tools and guide the organization of graded curricula.

### 🚀 Try It Yourself (No Coding Required!)
You do not need to install any software or know how to code to see this project in action. Click the badge below to open our interactive **Google Colab Notebook**.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WGqLul_ENM3dytF-7V7bTtYlT36GFNb5?usp=sharing)

---

## 💻 For Computational Researchers & Data Scientists (Reproducibility)
For NLP practitioners, this repository provides a reproducible pipeline for evaluating highly imbalanced, ordinal classification tasks in specialized historical text domains.

### 1. Dataset Information (*RhetoriScale*)
The repository utilizes the `RhetoriScale` dataset derived from the Burhan Classical Arabic rhetorical resource.
* **Total Instances:** 965 expertly annotated rhetorical segments.
* **Labels (Processing Effort):** 
  * `0` (Low Effort): Highly conventional, transparent figurative language (4.66%).
  * `1` (Medium Effort): Moderate inferential bridging required (79.17%).
  * `2` (High Effort): Highly dense, polysemous metaphors requiring deep exploration (16.17%).
* **Data Preprocessing:** Before tokenization, text undergoes standard preprocessing using the `PyArabic` library (removal of diacritics and Alef normalization) to ensure consistency across vocabularies.

### 2. Code Information
The codebase is modularly structured:
- `pipeline.py`: Executes the Stratified 5-Fold Cross-Validation, splits inner subsets, computes cost-sensitive weights, and manages Out-Of-Fold evaluations.
- `losses.py`: Implements the `OrdinalCrossEntropyLoss`, penalizing predictions based on ordinal distance.
- `custom_trainer.py`: HuggingFace `Trainer` override to inject the ordinal loss.
- `data_loader.py`: Handles fetching and JSON parsing using `json-repair`.
- `metrics.py`: Computes Accuracy, Macro-F1, QWK, and MAE.
- `stats.py` & `visualization.py`: Generates the Wilcoxon test, confusion matrices, and error severity distributions.

### 3. Requirements
To replicate the environment locally, ensure the following dependencies are installed (Python 3.8+ and a CUDA-enabled GPU are recommended):
```text
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.2.2
pandas>=2.0.0
numpy>=1.24.0
json-repair>=0.10.0
pyarabic>=0.6.15
matplotlib>=3.7.0
seaborn>=0.12.2
scipy>=1.10.0
tabulate>=0.9.0
```

### 4. Usage Instructions
You can run the provided scripts or use the Google Colab environment.
**Step 1: Load Data**
```python
from src.data_loader import load_data
df, label_encoder = load_data()
```
**Step 2: Train Model (Distance-Aware Ordinal Loss)**
```python
from src.pipeline import run_5fold_cv
MODEL_NAME = "UBC-NLP/ARBERT"
metrics, fold_results, preds, true, probs = run_5fold_cv(model_name=MODEL_NAME, df=df, use_ordinal=True)
```
**Step 3: Ablation Study (Standard Categorical CE)**
```python
metrics_baseline = run_5fold_cv(model_name=MODEL_NAME, df=df, use_ordinal=False)
```

### 5. Methodology Summary
1. **Data Partitioning:** Stratified 5-Fold CV scheme. Within each outer fold, an inner 85/15 split is performed for internal validation.
2. **Cost-Sensitive Weighting:** Inverse-frequency class weights are computed *exclusively* from the inner training subset to prevent data leakage.
3. **Distance-Aware Optimization:** The training objective minimizes a weighted categorical cross-entropy loss augmented by a distance penalty term.
4. **Checkpoint Selection:** The checkpoint achieving the highest Quadratic Weighted Kappa (QWK) on the internal validation set is selected for final testing on the blind outer fold.

---



This dataset and codebase are released under the **Creative Commons Attribution 4.0 International (CC-BY 4.0)** license. It is freely available for academic and non-commercial research purposes.

