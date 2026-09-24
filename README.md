# 📖 Rhetorix: Ordinal Learning for Cognitive Processing Effort in Classical Arabic Rhetoric

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WGqLul_ENM3dytF-7V7bTtYlT36GFNb5?usp=sharing)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project bridges the gap between **Deep Learning (NLP)** and the **Digital Humanities**, specifically targeting Classical Arabic and Qur'anic rhetoric. We teach artificial intelligence not just to *detect* metaphors, but to predict the *processing effort* required to interpret them, modeling this inferential burden as an expert-derived graded scale.

---

## 🏛️ For Humanities, Literature, and Linguistics Scholars
Are you a researcher in **Literary Theory, Applied Linguistics, Rhetoric, or Education**? This project is built with your theories in mind. 

We recognize that language is not just zeros and ones. By integrating concepts like **Relevance Theory** and **Cognitive Pragmatics**, this computational framework offers tangible tools for your research:

* 📚 **For Literary Critics & Theorists:** Explore how abstract theories of meaning-making (like graded salience and inferential burden) can be computationally modeled and validated across texts.
* 🖋️ **For Rhetoric & Arabic Scholars:** Move beyond the simple binary question of "Is this a metaphor?" to a deeper understanding of textual density and polysemy in Classical Arabic.
* 🎓 **For Applied Linguists & Educators:** Explore how these processing-effort models can inform future text readability tools, guide the organization of graded curricula for Arabic learners, and support advanced language proficiency assessments.

---

## 🚀 Try It Yourself (No Coding Required!)
You do not need to install any software or know how to code to see this project in action. We have prepared an interactive environment for you.

Click the badge below to open our **Google Colab Notebook**. This cloud-based tool allows you to run our experiment, explore the dataset, and see how the AI predicts cognitive effort step-by-step.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WGqLul_ENM3dytF-7V7bTtYlT36GFNb5?usp=sharing)

**How to use it:**
1. Click the link above.
2. Sign in with any Google account.
3. Read through the explanations and click the "Play" `▶` button on each cell to run the code.
4. Explore the `RhetoriScale` dataset and see the differences between *Low*, *Medium*, and *High* cognitive effort sentences.

---

## 💻 For Computational Researchers & Data Scientists
For NLP practitioners, this repository provides a reproducible pipeline for evaluating highly imbalanced, ordinal classification tasks in specialized historical text domains.

### The Technical Challenge
Traditional nominal classification (like Categorical Cross-Entropy) treats all errors equally. However, cognitive effort is a graded scale (`Low < Medium < High`). Predicting `High` when the true label is `Low` is a much worse error than predicting `Medium`. 

### Our Solution
We implement a **Distance-Aware Ordinal Loss Function** that penalizes the transformer model based on the geometric distance of its errors. Integrated with models like **ARBERT** and evaluated via a strict source-grouped Stratified 5-Fold Cross-Validation protocol (preventing data leakage), our framework effectively eliminates Severity-2 errors and significantly improves the Quadratic Weighted Kappa (QWK) score compared to standard categorical baselines.



---

## 📊 The Dataset: *RhetoriScale*
The dataset consists of Classical Arabic rhetorical segments expertly annotated into three ordinal levels of interpretive burden:
* **0 (Low Effort):** Highly conventional, transparent figurative language.
* **1 (Medium Effort):** Moderate inferential bridging required.
* **2 (High Effort):** Highly dense, polysemous, or historically opaque metaphors requiring deep contextual exploration.

*(Note: The full dataset is handled in accordance with the paper's data availability statement to protect the integrity of the expert annotations).*

---

## 📬 Contact
For technical inquiries regarding the code or models, please open an issue in this repository. For academic and theoretical collaborations, please reach out to the corresponding authors via the emails provided in the paper.
```

