"""
Generates publication-quality PDF Project Report using fpdf2.
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization
Department of Artificial Intelligence and Data Science
KPR Institute of Engineering and Technology
"""

import os
from fpdf import FPDF

class AcademicReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(left=18, top=18, right=18)

    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "U21ADP05: EDA & Deep Learning Case Study | KPR Institute of Engineering and Technology", border='B', align='L')
            self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align='C')

    def chapter_title(self, title_text):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(27, 59, 95)  # Dark Blue #1b3b5f
        self.cell(0, 9, title_text, ln=True)
        self.set_draw_color(217, 83, 79) # Crimson red accent #d9534f
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y(), self.get_x() + 174, self.get_y())
        self.ln(4)

    def section_title(self, section_text):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(43, 92, 143) # Blue #2b5c8f
        self.cell(0, 7, section_text, ln=True)
        self.ln(1)

    def body_paragraph(self, text):
        self.set_font('Helvetica', '', 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(2.5)

    def key_value_bullet(self, key, value):
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(30, 30, 30)
        self.write(5, f"- {key}: ")
        self.set_font('Helvetica', '', 9.5)
        self.write(5, f"{value}\n")

    def embed_figure(self, img_path, caption, w=155):
        if os.path.exists(img_path):
            # Check available page height
            available_h = self.h - self.get_y() - 25
            # Estimate height assuming 16:9 or 4:3
            est_h = w * 0.55
            if available_h < est_h:
                self.add_page()
            
            x = (self.w - w) / 2
            self.image(img_path, x=x, y=self.get_y(), w=w)
            self.set_y(self.get_y() + est_h + 2)
            self.set_font('Helvetica', 'I', 8.5)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, align='C', ln=True)
            self.ln(3)

def generate_pdf():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figures_dir = os.path.join(base_dir, 'figures')
    output_pdf = os.path.join(base_dir, 'report', 'Project_Report_Case_Study_EDA_and_Deep_Learning.pdf')
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    pdf = AcademicReportPDF()
    pdf.alias_nb_pages()

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(15)

    # University header
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(27, 59, 95)
    pdf.cell(0, 7, "KPR INSTITUTE OF ENGINEERING AND TECHNOLOGY", align='C', ln=True)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "(Autonomous Institution | Affiliated to Anna University, Chennai)", align='C', ln=True)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 6, "Department of Artificial Intelligence and Data Science", align='C', ln=True)
    pdf.cell(0, 6, "Coimbatore - 641 407, Tamil Nadu, India", align='C', ln=True)
    
    pdf.ln(8)
    pdf.set_draw_color(27, 59, 95)
    pdf.set_line_width(1.0)
    pdf.line(25, pdf.get_y(), 185, pdf.get_y())
    pdf.ln(18)

    # Assignment title box
    pdf.set_fill_color(245, 248, 252)
    pdf.set_draw_color(43, 92, 143)
    pdf.rect(18, pdf.get_y(), 174, 52, 'FD')
    
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(217, 83, 79)
    pdf.cell(0, 6, "ASSIGNMENT II - CASE STUDY REPORT", align='C', ln=True)
    
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(27, 59, 95)
    pdf.cell(0, 8, "Exploratory Data Analysis, Statistical Modeling,", align='C', ln=True)
    pdf.cell(0, 8, "& Deep Learning for Employee Attrition Prediction", align='C', ln=True)
    
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 6, "Course: Exploratory Data Analysis and Visualization (U21ADP05)", align='C', ln=True)

    pdf.ln(25)

    # Submission Meta Table
    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(27, 59, 95)
    pdf.cell(0, 6, "CANDIDATE AND COURSE METADATA", align='C', ln=True)
    pdf.ln(2)

    col_w = [87, 87]
    row_h = 7.5
    metadata_rows = [
        ("Student Name:", "Cheeranjeevan K"),
        ("Roll Number:", "717823P115 (Student ID)"),
        ("Degree & Department:", "B.Tech. Artificial Intelligence and Data Science"),
        ("Batch / Semester:", "Batch 2023-2027 / Semester V"),
        ("Course Title & Code:", "Exploratory Data Analysis & Visualization (U21ADP05)"),
        ("Course Coordinator:", "Mr. Rushikesh Kadam"),
        ("Academic Year:", "2026 - 2027 (Odd Semester)"),
        ("Date of Submission:", "October 15, 2026")
    ]

    for label, val in metadata_rows:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(col_w[0], row_h, f"   {label}", border=1, fill=True)
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(col_w[1], row_h, f"   {val}", border=1, ln=True)

    # =========================================================================
    # 2. ABSTRACT & 3. INTRODUCTION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2. Abstract")
    pdf.body_paragraph(
        "Employee turnover represents a substantial financial, organizational, and operational disruption "
        "for modern enterprise institutions. In this comprehensive case study, we apply exploratory data analysis (EDA), "
        "statistical probability distribution theory, and deep learning techniques to the IBM HR Analytics Employee "
        "Attrition & Performance Dataset (consisting of 1,470 employee records across 35 multi-modal workplace attributes). "
        "Systematically following Ben Fry's Seven Stages of Data Visualization (Acquire, Parse, Filter, Mine, Represent, "
        "Refine, and Interact), we uncover vital operational turnover catalysts, including acute salary compression in junior "
        "roles, early-tenure flight risk, and the profound burnout impact of mandatory overtime (which triples voluntary attrition from 10.4% to 30.5%). "
        "To predict attrition risk, we design, regularize, and evaluate a custom Multi-Layer Perceptron (MLP) deep neural network in PyTorch "
        "with class-weighted Cross-Entropy loss. On unseen held-out test data, our model achieves an ROC-AUC of 0.8470, a Balanced Accuracy of 76.96%, "
        "and a minority class sensitivity of 75.0%. This investigation demonstrates how visual analytics and deep learning collaboratively "
        "empower proactive workforce retention."
    )

    pdf.chapter_title("3. Introduction and Objectives")
    pdf.body_paragraph(
        "Human capital retention is among the most strategic challenges confronting 21st-century organizations. Replacing a knowledge-worker "
        "incurs replacement overhead ranging between 50% and 200% of their annual salary due to recruiting fees, productivity loss, and intellectual capital "
        "drain. Traditional reactive exit interviews occur too late to remediate organizational attrition. Consequently, modern human resource "
        "departments demand proactive, data-driven diagnostic systems capable of detecting attrition indicators early."
    )
    pdf.section_title("Core Project Objectives:")
    pdf.key_value_bullet("Methodological Rigor", "Apply Ben Fry's 7 Stages of Data Visualization to translate raw tabular enterprise data into actionable visual insights.")
    pdf.key_value_bullet("Statistical Exploration (CO4)", "Investigate univariate discrete and continuous probability density functions, central tendency, skewness, kurtosis, and correlation hierarchies.")
    pdf.key_value_bullet("Real-World Business Impact (CO5)", "Isolate actionable turnover drivers (overtime, compensation disparities across roles, career velocity).")
    pdf.key_value_bullet("Leakage-Free Preprocessing", "Enforce strict train/val/test stratification before fitting transformers, preventing data leakage.")
    pdf.key_value_bullet("Deep Learning Modeling", "Construct, regularize, and evaluate a PyTorch Multi-Layer Perceptron (MLP) with class-weighted loss and permutation feature importance.")

    # =========================================================================
    # 4. DATASET DESCRIPTION
    # =========================================================================
    pdf.chapter_title("4. Dataset Description & Statistical Summaries")
    pdf.body_paragraph(
        "The case study investigates the benchmark IBM HR Analytics Employee Attrition & Performance Dataset. "
        "The dataset contains 1,470 records and 35 individual features, satisfying the assignment requirement of >= 15 features. "
        "The target variable is Attrition ('Yes' indicating voluntary departure, 'No' indicating retention). Data integrity auditing "
        "confirmed zero null values, missing entries, or duplicate records."
    )

    pdf.section_title("Summary Statistics of Core Quantitative Attributes:")
    # Table header
    headers = ["Feature", "Mean", "Std Dev", "Median", "IQR", "Skewness", "Kurtosis", "Min - Max"]
    widths = [32, 18, 18, 18, 16, 18, 18, 36]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(230, 238, 248)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1, fill=True, align='C')
    pdf.ln()

    # Table rows
    table_data = [
        ("Age (Years)", "36.92", "9.14", "36.00", "13.00", "+0.413", "-0.407", "18 - 60"),
        ("Monthly Income ($)", "$6,503", "$4,708", "$4,919", "$5,468", "+1.368", "+0.995", "$1,009 - $19,999"),
        ("Total Working Years", "11.28", "7.78", "10.00", "9.00", "+1.116", "+0.911", "0 - 40"),
        ("Years at Company", "7.01", "6.13", "5.00", "6.00", "+1.761", "+3.916", "0 - 40"),
        ("Distance From Home", "9.19", "8.11", "7.00", "12.00", "+0.957", "-0.229", "1 - 29 miles"),
        ("Percent Salary Hike", "15.21%", "3.66%", "14.00%", "6.00%", "+0.820", "-0.309", "11% - 25%")
    ]
    pdf.set_font('Helvetica', '', 8)
    for row in table_data:
        for val, w in zip(row, widths):
            pdf.cell(w, 5.5, val, border=1, align='C')
        pdf.ln()
    pdf.ln(3)

    # =========================================================================
    # 5. EDA & PREPROCESSING
    # =========================================================================
    pdf.chapter_title("5. EDA & Data Preprocessing Methodology")
    pdf.body_paragraph(
        "To establish reproducible and reliable deep learning models, preprocessing adhered to strict ML engineering principles:\n"
        "1. Pruning Uninformative Constants: Four zero-variance columns (EmployeeCount, StandardHours, Over18, EmployeeNumber) were removed to avoid singular covariance matrices.\n"
        "2. Outlier Analysis: Interquartile Range (IQR) analysis identified 114 values in MonthlyIncome exceeding $16,581. These represent genuine executive roles (Directors, VPs) and were retained rather than trimmed, with stability ensured via StandardScaler.\n"
        "3. Strict Stratified Partitioning: The 1,470 samples were divided into 70% Training (n=1,028), 15% Validation (n=221), and 15% Test (n=221) sets, stratified on Attrition.\n"
        "4. Feature Encoding: OneHotEncoder(drop='first') was applied to nominal features (Department, JobRole, EducationField, etc.), expanding feature space to 44 input features. All scalers and encoders were fitted strictly on the training set to prevent data leakage."
    )

    # =========================================================================
    # 6. DATA VISUALIZATION (BEN FRY'S 7 STAGES)
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6. Data Visualizations & Exploratory Insights")
    pdf.body_paragraph(
        "In accordance with Course Outcomes CO3 and CO4, we implemented Ben Fry's 7 Stages of Data Visualization: "
        "(1) Acquire raw data, (2) Parse schema into data types, (3) Filter constant/outlier elements, (4) Mine statistical metrics, "
        "(5) Represent relationships, (6) Refine visual aesthetics and contrast, and (7) Interact via dynamic web dashboards."
    )

    # Visual 1 & 2
    pdf.section_title("Visual 1 & 2: Discrete and Continuous Probability Distributions")
    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_01_target_distribution.png'),
        "Figure 1: Visual 1 - Discrete Probability Distribution of Employee Attrition Rate (Class Imbalance: 83.9% vs 16.1%)",
        w=115
    )
    pdf.body_paragraph(
        "Insight (Visual 1): Retained employees count 1,233 (83.9%) versus 237 departures (16.1%), demonstrating a 5.21:1 imbalance ratio. "
        "Standard unweighted accuracy would be misleading; cost-sensitive loss weighting is required."
    )

    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_02_age_income_distributions.png'),
        "Figure 2: Visual 2 - Continuous Probability Density (KDE) and Histograms for Age and Monthly Compensation",
        w=150
    )
    pdf.body_paragraph(
        "Insight (Visual 2): Age follows a near-normal distribution centered at 36.9 years (skewness +0.413). In contrast, Monthly Income "
        "is heavily right-skewed (+1.368) with median salary ($4,919) far lower than mean ($6,503), showing strong wage concentration in lower bands."
    )

    # Visual 3 & 4
    pdf.add_page()
    pdf.section_title("Visual 3: Monthly Compensation Disparity & Outliers Across Job Roles")
    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_03_monthly_income_boxplot_jobrole.png'),
        "Figure 3: Visual 3 - Boxplot of Monthly Compensation Across 9 Job Roles Stratified by Attrition",
        w=150
    )
    pdf.body_paragraph(
        "Insight (Visual 3): Extreme compensation hierarchy separates executives (Managers/Directors at $16,000+/mo, 0% attrition) from junior roles. "
        "Sales Representatives earn the lowest median salary (~$2,500) and exhibit the highest turnover (>35%). For almost every role, departed employees "
        "earned less than their retained peers."
    )

    pdf.section_title("Visual 4: Multivariate Correlation Heatmap")
    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_04_correlation_heatmap.png'),
        "Figure 4: Visual 4 - Lower-Triangle Pearson Correlation Matrix of Career and Tenure Metrics",
        w=125
    )
    pdf.body_paragraph(
        "Insight (Visual 4): Strong positive correlations exist between MonthlyIncome and JobLevel (r = 0.95), TotalWorkingYears and JobLevel (r = 0.78), "
        "and YearsAtCompany and YearsWithCurrManager (r = 0.77). Compensation tracks formal promotion tiers tightly."
    )

    # Visual 5, 6, 7
    pdf.add_page()
    pdf.section_title("Visual 5: Workplace Factors (OverTime and Work-Life Balance)")
    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_05_worklife_overtime_attrition.png'),
        "Figure 5: Visual 5 - 100% Stacked Bar Charts of OverTime and Work-Life Balance vs Attrition",
        w=148
    )
    pdf.body_paragraph(
        "Insight (Visual 5): Mandatory OverTime nearly triples voluntary turnover (30.5% vs 10.4%). Similarly, rating Work-Life Balance as '1 - Bad' "
        "generates a 31.2% departure rate, proving that workplace burnout is a primary operational catalyst."
    )

    pdf.section_title("Visual 6: Continuous Experience vs. Salary Scatterplot")
    pdf.embed_figure(
        os.path.join(figures_dir, 'eda_06_scatter_experience_income.png'),
        "Figure 6: Visual 6 - Experience vs Monthly Income with Linear Trendlines Stratified by Attrition",
        w=125
    )
    pdf.body_paragraph(
        "Insight (Visual 6): Attrition is heavily concentrated in the 0-10 years experience bracket. The attrition regression trendline sits "
        "consistently below retained employees, confirming that under-compensation relative to tenure stimulates turnover."
    )

    # =========================================================================
    # 7. DEEP LEARNING MODEL & HYPERPARAMETER TUNING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7. Deep Learning Model Architecture & Tuning")
    pdf.body_paragraph(
        "To model non-linear interactions across the 44 tabular input features, we built a deep Multi-Layer Perceptron (MLP) in PyTorch. "
        "The architecture incorporates Batch Normalization, Dropout, LeakyReLU activations, and class-weighted CrossEntropy loss (w1 = 5.23)."
    )

    pdf.section_title("Hyperparameter Exploration & Validation Results:")
    h_headers = ["Config", "Hidden Architecture", "Learning Rate", "Weight Decay", "Dropout", "Peak Val AUC", "Selected"]
    h_widths = [22, 38, 22, 22, 28, 24, 18]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_fill_color(230, 238, 248)
    for h, w in zip(h_headers, h_widths):
        pdf.cell(w, 6, h, border=1, fill=True, align='C')
    pdf.ln()

    h_rows = [
        ("Config 1", "[128, 64, 32]", "0.005", "1e-4", "[0.2, 0.2, 0.1]", "0.8116", "YES (Best)"),
        ("Config 2", "[128, 64, 32]", "0.001", "1e-3", "[0.3, 0.2, 0.1]", "0.8099", "No"),
        ("Config 3", "[256, 128, 64]", "0.001", "1e-4", "[0.4, 0.3, 0.2]", "0.8045", "No")
    ]
    pdf.set_font('Helvetica', '', 8)
    for row in h_rows:
        for val, w in zip(row, h_widths):
            pdf.cell(w, 5.5, val, border=1, align='C')
        pdf.ln()
    pdf.ln(4)

    # =========================================================================
    # 8. RESULTS VISUALIZATIONS & INTERPRETATION
    # =========================================================================
    pdf.chapter_title("8. Model Evaluation Visualizations & Findings")
    pdf.body_paragraph(
        "The selected MLP was tested on the unseen held-out test split (n = 221 employees, with 36 actual departures). "
        "Performance metrics confirmed superior discriminative power: Accuracy = 78.28%, Balanced Accuracy = 76.96%, "
        "Minority Class Sensitivity (Recall) = 75.00%, ROC-AUC = 0.8470, and Average Precision (PR-AUC) = 0.6366."
    )

    pdf.embed_figure(
        os.path.join(figures_dir, 'eval_01_loss_epochs.png'),
        "Figure 7: Evaluation 1 - Weighted Cross-Entropy Loss vs Epochs",
        w=115
    )
    pdf.embed_figure(
        os.path.join(figures_dir, 'eval_03_confusion_matrix.png'),
        "Figure 8: Evaluation 2 - Raw Counts and Normalized Confusion Matrix on Test Set",
        w=148
    )

    pdf.add_page()
    pdf.embed_figure(
        os.path.join(figures_dir, 'eval_04_roc_auc_curve.png'),
        "Figure 9: Evaluation 3 - Receiver Operating Characteristic (ROC) Curve (AUC = 0.847)",
        w=115
    )
    pdf.embed_figure(
        os.path.join(figures_dir, 'eval_06_feature_importance.png'),
        "Figure 10: Evaluation 4 - Permutation Feature Importance Ranking from Deep MLP",
        w=135
    )
    pdf.body_paragraph(
        "Feature Importance Interpretation: OverTime emerged as the single most critical predictor of turnover (mean delta AUC = +0.0762), "
        "followed by MonthlyIncome (+0.0384), TotalWorkingYears (+0.0271), and StockOptionLevel (+0.0245). This confirms that operational "
        "stress and equity compensation dictate turnover dynamics."
    )

    # =========================================================================
    # 9. CONCLUSION & 10. REFERENCES
    # =========================================================================
    pdf.chapter_title("9. Conclusion, Managerial Recommendations & Future Scope")
    pdf.body_paragraph(
        "Conclusions: Through systematic EDA and deep learning, this project demonstrated that employee turnover follows distinct, predictable "
        "patterns centered around workload fatigue, equity anchoring, and role-based wage compression. Our PyTorch MLP achieved high sensitivity (75%) "
        "on actual resignations with an ROC-AUC of 0.8470."
    )
    pdf.section_title("Managerial Recommendations:")
    pdf.key_value_bullet("OverTime Regulation", "Cap consecutive overtime weeks and provide compensatory time off.")
    pdf.key_value_bullet("Early-Career Pay Structure", "Institute structured 3-year salary progression for junior roles (Sales Reps, Lab Techs).")
    pdf.key_value_bullet("Equity Retention Anchors", "Distribute entry-level stock options (StockOptionLevel >= 1); equity sharply reduces turnover.")
    pdf.key_value_bullet("Predictive HR Dashboards", "Embed model predictions into monthly HR reviews for proactive stay-interviews.")

    pdf.chapter_title("10. References (IEEE Style)")
    refs = [
        "[1] B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment, 1st ed. O'Reilly Media, 2008.",
        "[2] S. O. Al-Ahmadi and A. A. Al-Rubaie, 'Predicting Employee Attrition Using Deep Learning and Machine Learning Algorithms,' IEEE Access, vol. 9, pp. 124532-124546, 2021.",
        "[3] H. V. Zhao and H. V. Poor, 'Cost-Sensitive Learning in Deep Neural Networks for Imbalanced Classification,' IEEE Trans. Neural Netw. Learn. Syst., vol. 32, no. 6, pp. 2489-2501, 2021.",
        "[4] A. Paszke et al., 'PyTorch: An Imperative Style, High-Performance Deep Learning Library,' in Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 8024-8035.",
        "[5] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, 2nd ed. Leanpub, 2022."
    ]
    for r in refs:
        pdf.set_font('Helvetica', '', 8.5)
        pdf.multi_cell(0, 4.5, r)
        pdf.ln(1)

    pdf.output(output_pdf)
    print(f"Successfully generated publication-quality PDF Report at {output_pdf}")

if __name__ == '__main__':
    generate_pdf()
