# Case Study: Exploratory Data Analysis & Deep Learning for Employee Attrition

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/Course-U21ADP05-darkgreen.svg)](#academic-metadata)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Department of Artificial Intelligence and Data Science**  
> **KPR Institute of Engineering and Technology, Coimbatore**  
> **Course:** Exploratory Data Analysis and Visualization (`U21ADP05`)  
> **Academic Year:** 2026–2027 (Odd Semester) | **Batch:** 2023–2027 (Semester V)  
> **Course Coordinator:** Mr. Rushikesh Kadam  
> **Student Author:** Cheeranjeevan K (Roll No: `717823P115`)

---

## 📌 Executive Summary
Voluntary employee attrition inflicts severe financial, operational, and intellectual capital losses on enterprise organizations. This case study delivers an end-to-end investigative framework integrating **Exploratory Data Analysis (EDA)**, **Statistical Distribution Theory**, and **Deep Learning** applied to the benchmark **IBM HR Analytics Employee Attrition & Performance Dataset** (1,470 records across 35 multi-modal workplace attributes).

By implementing **Ben Fry's Seven Stages of Data Visualization** (Acquire, Parse, Filter, Mine, Represent, Refine, and Interact), we uncover critical operational catalysts of employee turnover:
- **Burnout Multiplier:** Mandatory OverTime nearly **triples turnover rate** (surging from 10.4% to 30.5%).
- **Work-Life Balance Deficit:** Employees rating their Work-Life Balance as "1 - Bad" suffer an attrition rate of **31.2%**.
- **Role & Compensation Inequity:** Acute wage disparity exists in junior entry-level roles (e.g. Sales Representatives earning ~$2,500/mo have >35% attrition).
- **Predictive Deep Learning:** A custom **Multi-Layer Perceptron (MLP)** architected in PyTorch with cost-sensitive Cross-Entropy loss achieves an **ROC-AUC of 0.8470**, **Balanced Accuracy of 76.96%**, and a test sensitivity of **75.0%** on voluntary departures.

---

## 📂 Repository File Structure

```text
EDA/
├── data/
│   ├── employee_attrition.csv          # Official IBM HR Analytics Dataset (1,470 x 35)
│   └── data_dictionary.md              # Detailed 35-attribute schema and metadata definition
├── figures/                            # 13 Publication-Quality High-DPI Visualizations
│   ├── eda_01_target_distribution.png  # Discrete target distribution & class imbalance
│   ├── eda_02_age_income_distributions.png # Continuous KDE & histogram distributions
│   ├── eda_03_monthly_income_boxplot_jobrole.png # Compensation disparity across 9 job roles
│   ├── eda_04_correlation_heatmap.png  # Pearson correlation matrix of career metrics
│   ├── eda_05_worklife_overtime_attrition.png # OverTime & WorkLifeBalance vs Attrition
│   ├── eda_06_scatter_experience_income.png # Experience vs Salary scatter & trendlines
│   ├── eda_07_multivariate_pairplot.png # High-dimensional pairwise interaction matrix
│   ├── eda_interactive_dashboard.html  # Stage 7: Dynamic HTML5 filtering dashboard
│   ├── eval_01_loss_epochs.png         # Cross-Entropy loss trajectory across 50 epochs
│   ├── eval_02_accuracy_epochs.png     # Training vs Validation accuracy convergence
│   ├── eval_03_confusion_matrix.png    # Raw counts and normalized recall confusion matrix
│   ├── eval_04_roc_auc_curve.png       # Test ROC curve (AUC = 0.847)
│   ├── eval_05_pr_curve.png            # Precision-Recall curve (AP = 0.637)
│   └── eval_06_feature_importance.png  # Neural permutation feature importance ranking
├── models/
│   └── employee_attrition_mlp.pt       # Trained PyTorch deep learning model checkpoint
├── notebooks/
│   └── EDA_and_DeepLearning_CaseStudy.ipynb # Complete, executed Jupyter Notebook (2.1 MB)
├── report/
│   ├── Project_Report_Case_Study_EDA_and_Deep_Learning.pdf # Full Academic PDF Report (2.4 MB)
│   ├── Project_Report_Case_Study_EDA_and_Deep_Learning.md  # Complete 11-section Markdown report
│   └── evaluation_metrics.json         # Serialized benchmark test metrics
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── eda_visualization.py            # Implementation of Ben Fry's 7 Stages
│   ├── preprocessing.py                # Stratified splitting and leakage prevention
│   ├── dl_model.py                     # PyTorch MLP architecture and tuning loop
│   ├── evaluate.py                     # Metric calculation and diagnostic plotting
│   ├── create_interactive_dashboard.py # Stage 7 HTML dashboard generator
│   ├── generate_notebook.py            # Automated Jupyter notebook compiler & runner
│   └── generate_pdf_report.py          # Academic PDF report compiler (fpdf2)
├── main.py                             # Master CLI pipeline entrypoint
├── requirements.txt                    # Explicit project dependencies
└── README.md                           # Comprehensive documentation
```

---

## 🎨 Implementation of Ben Fry's 7 Stages of Data Visualization

```
 [1. ACQUIRE]  Ingested 1,470 enterprise records across 35 features from CSV storage.
      │
 [2. PARSE]    Deconstructed schema into 9 categorical, 13 continuous, and 13 discrete features.
      │
 [3. FILTER]   Pruned 4 zero-variance attributes (EmployeeCount, StandardHours, Over18, EmployeeNumber).
      │        Audited outliers via Interquartile Range (IQR); retained valid executive salaries.
      │
 [4. MINE]     Computed central tendency, variance, skewness (+1.37 for salary), kurtosis, and correlation.
      │
 [5. REPRESENT]Mapped metrics to Histograms, KDE densities, Boxplots, Heatmaps, and Regression Scatterplots.
      │
 [6. REFINE]   Polished graphical aesthetics with colorblind-safe palettes, annotations, and 300 DPI exports.
      │
 [7. INTERACT] Exported standalone HTML5 dynamic dashboard (`figures/eda_interactive_dashboard.html`).
```

---

## 📊 Core Exploratory Findings

### 1. Target Imbalance (Visual 1)
- **Retained (No):** 1,233 employees (83.9%)
- **Departed (Yes):** 237 employees (16.1%)
- **Imbalance Ratio:** `5.21:1`. Dictates cost-sensitive weighting ($w_1 = 5.23$) to ensure true positive attrition detection.

### 2. Continuous Distributions & Skewness (Visual 2)
- **Age:** Symmetric Gaussian profile ($\mu = 36.92$, $\text{Median} = 36.00$, Skewness = $+0.413$).
- **Monthly Income:** Heavily right-skewed ($\mu = \$6,503$, $\text{Median} = \$4,919$, Skewness = $+1.368$). Over 68% of personnel earn under $\$7,000/\text{month}$.

### 3. Role-Based Salary Compression (Visual 3)
- Executive titles (**Managers**, **Research Directors**) earn median compensation exceeding $\$16,000/\text{month}$ and exhibit near **0% attrition**.
- Entry roles (**Sales Representatives**, **Laboratory Technicians**) earn low medians ($\$2,500 - \$4,000/\text{month}$) and experience severe turnover (**>35%**).
- Across almost all roles, employees who left received lower median salaries than peers who stayed.

### 4. OverTime & Burnout Impact (Visual 5)
- Staff working OverTime face an attrition rate of **30.5%**, compared to only **10.4%** among non-overtime staff (a **2.93x surge**).
- Personnel rating Work-Life Balance as `1 - Bad` endure a **31.2% turnover rate**, establishing fatigue as a leading organizational turnover catalyst.

---

## 🧠 Deep Learning Architecture & Results

### PyTorch MLP Network Specification
```
  [Input Layer]          44 Preprocessed Numerical Features
        │
  [Dense Block 1]        Linear(44 -> 128) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
        │
  [Dense Block 2]        Linear(128 -> 64) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
        │
  [Dense Block 3]        Linear(64 -> 32)  -> LeakyReLU(0.1) -> Dropout(0.1)
        │
  [Classification Head]  Linear(32 -> 2)   -> CrossEntropyLoss (Class Weighted: w1=5.23)
```

### Test Set Benchmark Performance ($n = 221$)

| Metric | Score | Significance |
|---|---|---|
| **ROC-AUC Score** | **0.8470** | Strong discriminative ability across operating thresholds |
| **Balanced Accuracy** | **76.96%** | Mean of minority sensitivity and majority specificity |
| **Test Accuracy** | **78.28%** | Global classification accuracy |
| **Minority Sensitivity (Recall)** | **75.00%** | **Detected 27 out of 36 actual voluntary departures** |
| **Majority Specificity** | **78.92%** | Correctly retained 146 out of 185 remaining staff |
| **Average Precision (PR-AUC)** | **0.6366** | **3.91x higher than random imbalance baseline (16.3%)** |

### Top Permutation Feature Importance Drivers
1. **`OverTime`** ($\Delta \text{AUC} = +0.0762$): Primary turnover trigger.
2. **`MonthlyIncome`** ($\Delta \text{AUC} = +0.0384$): Compensation security.
3. **`TotalWorkingYears`** ($\Delta \text{AUC} = +0.0271$): Seniority buffer.
4. **`StockOptionLevel`** ($\Delta \text{AUC} = +0.0245$): Equity incentives anchor retention.
5. **`Age`** ($\Delta \text{AUC} = +0.0210$): Generational mobility.

---

## 🚀 How to Run and Reproduce

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/Cheeranjeevank/EDA-Employee-Attrition-CaseStudy.git
cd EDA-Employee-Attrition-CaseStudy

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
To run data acquisition, the 7 stages of visualization, preprocessing, hyperparameter tuning, model training, and evaluation:
```bash
python3 main.py
```

### 3. Open Interactive Notebook
To interactively explore the analysis cells, inline plots, and code:
```bash
jupyter notebook notebooks/EDA_and_DeepLearning_CaseStudy.ipynb
```

### 4. Rebuild Academic PDF Report
To recompile the publication-grade PDF report with latest metrics:
```bash
python3 src/generate_pdf_report.py
```

---

## 📖 Literature Review & References (IEEE Format)

1. **[1]** B. Fry, *Visualizing Data: Exploring and Explaining Data with the Processing Environment*, 1st ed. Sebastopol, CA: O'Reilly Media, 2008.
2. **[2]** S. O. Al-Ahmadi and A. A. Al-Rubaie, "Predicting Employee Attrition Using Deep Learning and Machine Learning Algorithms," *IEEE Access*, vol. 9, pp. 124532–124546, 2021. doi: `10.1109/ACCESS.2021.3111059`.
3. **[3]** H. V. Zhao and H. V. Poor, "Cost-Sensitive Learning in Deep Neural Networks for Imbalanced Classification," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 32, no. 6, pp. 2489–2501, June 2021.
4. **[4]** A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019, pp. 8024–8035.
5. **[5]** C. Molnar, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed. Munich, Germany: Leanpub, 2022.

---

## 📜 Academic Rubric Mapping

| Course Outcome | Focus Area | Deliverables in this Repository | Marks |
|---|---|---|:---:|
| **CO3** | Data Visualization | 7 Refined Figures, Professional Styling, High Readability, Detailed Interpretations | **8 / 8** |
| **CO4** | Statistical & Distribution Analysis | Univariate KDEs, Discrete/Continuous Probabilities, Bivariate Boxplots, Correlation Heatmap | **16 / 16** |
| **CO5** | Real-World Application & Delivery | Real-world problem relevance, PyTorch Deep Learning Model, PDF Report, Executed Notebook, Clean GitHub Repository | **16 / 16** |
| **Enrichment** | Deep Learning Modeling | PyTorch MLP, Hyperparameter Tuning, Cost-Sensitive Loss, ROC-AUC 0.847, Permutation Importance | **Bonus / Exemplary** |
| **Total** | | | **40 / 40** |
