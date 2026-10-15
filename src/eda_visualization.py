"""
Exploratory Data Analysis and Visualization Module
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization
Academic Year: 2026-2027 (Batch 2023-2027)

This module systematically executes Ben Fry's 7 Stages of Data Visualization:
1. ACQUIRE: Ingest real-world raw dataset
2. PARSE: Provide structural typing, schema organization, discrete/continuous breakdown
3. FILTER: Remove non-informative constants, inspect outliers and anomalies
4. MINE: Extract statistical summaries, skewness, kurtosis, and correlation matrices
5. REPRESENT: Render statistical visualizations (Histograms, KDEs, Boxplots, Heatmaps, Scatterplots)
6. REFINE: Polish graphical design with curated palettes, typography, annotations, high-DPI export
7. INTERACT: Generate interactive visual representations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eaeaea'
plt.rcParams['grid.linestyle'] = '--'

PALETTE = {'No': '#2b5c8f', 'Yes': '#d9534f'}
COLORS = ['#2b5c8f', '#d9534f', '#41b6c4', '#225ea8', '#fdae61']

def stage_1_acquire(filepath: str) -> pd.DataFrame:
    """Stage 1: Acquire - Load raw data from source."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[Stage 1: ACQUIRE] Loaded dataset from {filepath} with shape {df.shape}")
    return df

def stage_2_parse(df: pd.DataFrame) -> dict:
    """Stage 2: Parse - Structure, classify features, and identify data types."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Classify discrete vs continuous numerical
    discrete_num = [c for c in numerical_cols if df[c].nunique() <= 10]
    continuous_num = [c for c in numerical_cols if df[c].nunique() > 10]
    
    summary = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'categorical': categorical_cols,
        'discrete_numerical': discrete_num,
        'continuous_numerical': continuous_num,
        'missing_values': df.isnull().sum().to_dict()
    }
    print(f"[Stage 2: PARSE] Categorical: {len(categorical_cols)}, Continuous: {len(continuous_num)}, Discrete: {len(discrete_num)}")
    return summary

def stage_3_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: Filter - Detect zero-variance/constant attributes and identify outliers."""
    # Constant columns in IBM HR dataset: EmployeeCount (all 1), Over18 (all Y), StandardHours (all 80), EmployeeNumber (arbitrary ID)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    id_cols = ['EmployeeNumber'] if 'EmployeeNumber' in df.columns else []
    to_drop = list(set(constant_cols + id_cols))
    
    df_clean = df.drop(columns=to_drop)
    print(f"[Stage 3: FILTER] Dropped zero-variance / ID columns: {to_drop}. Retained shape: {df_clean.shape}")
    
    # Outlier detection on key continuous columns using IQR
    outlier_summary = {}
    for col in ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome']:
        if col in df_clean.columns:
            q25, q75 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_summary[col] = {
                'iqr': iqr,
                'outlier_count': len(outliers),
                'outlier_percentage': round(len(outliers) / len(df_clean) * 100, 2)
            }
    print(f"[Stage 3: FILTER] Outlier audit via IQR: {outlier_summary}")
    return df_clean

def stage_4_mine(df: pd.DataFrame) -> dict:
    """Stage 4: Mine - Compute statistical summaries, distributions, skewness, and correlations."""
    stats_dict = {}
    for col in ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome']:
        if col in df.columns:
            series = df[col]
            stats_dict[col] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'median': float(series.median()),
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'min': float(series.min()),
                'max': float(series.max())
            }
    
    # Target distribution
    target_counts = df['Attrition'].value_counts()
    target_pct = df['Attrition'].value_counts(normalize=True) * 100
    stats_dict['target'] = {
        'counts': target_counts.to_dict(),
        'percentages': target_pct.to_dict()
    }
    print(f"[Stage 4: MINE] Mined statistical metrics. Target balance: {target_pct.to_dict()}")
    return stats_dict

def stage_5_and_6_represent_and_refine(df: pd.DataFrame, output_dir: str):
    """
    Stages 5 & 6: Represent & Refine - Generate and polish publication-quality figures.
    Generates 7 core visuals covering:
      - Discrete probability distribution (Univariate)
      - Continuous probability distribution (Univariate)
      - Bivariate distribution and outlier boxplots
      - Multivariate correlation heatmap
      - Grouped categorical factor interactions
      - Continuous joint relationship & regression scatterplot
      - High-dimensional pairplot interactions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Visual 1: Target Distribution & Class Imbalance (Discrete Distribution)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    counts = df['Attrition'].value_counts()
    pcts = df['Attrition'].value_counts(normalize=True) * 100
    labels = ['No (Stayed)', 'Yes (Departed)']
    bars = ax.bar(['No', 'Yes'], counts, color=['#2b5c8f', '#d9534f'], width=0.55, edgecolor='#1b3b5f', linewidth=1.2)
    
    for bar, count, pct in zip(bars, counts, pcts):
        height = bar.get_height()
        ax.annotate(f"{count:,} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#222222')
        
    ax.set_title("Visual 1: Target Variable Distribution (Employee Attrition Rate)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Attrition Status", fontsize=12, labelpad=10)
    ax.set_ylabel("Employee Count", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    v1_path = os.path.join(output_dir, "eda_01_target_distribution.png")
    fig.savefig(v1_path, dpi=300)
    plt.close(fig)
    print(f"[Visual 1] Saved target distribution to {v1_path}")

    # -------------------------------------------------------------
    # Visual 2: Continuous Probability Distributions (Age & Monthly Income)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Age distribution
    age = df['Age']
    sns.histplot(age, kde=True, ax=axes[0], color='#2b5c8f', bins=20, stat='density', edgecolor='white', alpha=0.6)
    axes[0].axvline(age.mean(), color='#d9534f', linestyle='--', linewidth=2, label=f'Mean: {age.mean():.1f}')
    axes[0].axvline(age.median(), color='#2ca02c', linestyle='-', linewidth=2, label=f'Median: {age.median():.1f}')
    axes[0].set_title(f"Age Distribution (Skewness: {stats.skew(age):.2f})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Employee Age (Years)", fontsize=11)
    axes[0].set_ylabel("Probability Density", fontsize=11)
    axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Monthly Income distribution
    income = df['MonthlyIncome']
    sns.histplot(income, kde=True, ax=axes[1], color='#d9534f', bins=25, stat='density', edgecolor='white', alpha=0.6)
    axes[1].axvline(income.mean(), color='#2b5c8f', linestyle='--', linewidth=2, label=f'Mean: ${income.mean():,.0f}')
    axes[1].axvline(income.median(), color='#2ca02c', linestyle='-', linewidth=2, label=f'Median: ${income.median():,.0f}')
    axes[1].set_title(f"Monthly Income Distribution (Skewness: {stats.skew(income):.2f})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Monthly Salary (USD)", fontsize=11)
    axes[1].set_ylabel("Probability Density", fontsize=11)
    axes[1].legend(frameon=True, facecolor='white', framealpha=0.9)
    
    fig.suptitle("Visual 2: Continuous Probability Distributions and Univariate Exploration", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    v2_path = os.path.join(output_dir, "eda_02_age_income_distributions.png")
    fig.savefig(v2_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visual 2] Saved continuous distributions to {v2_path}")

    # -------------------------------------------------------------
    # Visual 3: Bivariate Distribution & Outlier Boxplot (Monthly Income by Job Role)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    order = df.groupby('JobRole')['MonthlyIncome'].median().sort_values().index
    sns.boxplot(data=df, x='JobRole', y='MonthlyIncome', hue='Attrition',
                palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, order=order, ax=ax,
                fliersize=4, flierprops={'marker': 'o', 'markerfacecolor': '#d9534f', 'alpha': 0.6})
    
    ax.set_title("Visual 3: Monthly Income Distribution & Outliers across Job Roles by Attrition Status", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Job Role (Ordered by Median Compensation)", fontsize=12, labelpad=10)
    ax.set_ylabel("Monthly Income ($ USD)", fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    ax.legend(title='Attrition', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    v3_path = os.path.join(output_dir, "eda_03_monthly_income_boxplot_jobrole.png")
    fig.savefig(v3_path, dpi=300)
    plt.close(fig)
    print(f"[Visual 3] Saved salary boxplot to {v3_path}")

    # -------------------------------------------------------------
    # Visual 4: Multivariate Correlation Heatmap
    # -------------------------------------------------------------
    num_cols_for_corr = [
        'Age', 'MonthlyIncome', 'TotalWorkingYears', 'JobLevel', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'DistanceFromHome', 'PercentSalaryHike'
    ]
    corr_matrix = df[num_cols_for_corr].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.3, vmax=1.0, center=0,
                square=True, linewidths=0.6, cbar_kws={"shrink": 0.8},
                annot=True, fmt=".2f", annot_kws={"size": 9}, ax=ax)
    
    ax.set_title("Visual 4: Multivariate Correlation Heatmap of Career and Tenure Metrics", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    v4_path = os.path.join(output_dir, "eda_04_correlation_heatmap.png")
    fig.savefig(v4_path, dpi=300)
    plt.close(fig)
    print(f"[Visual 4] Saved correlation heatmap to {v4_path}")

    # -------------------------------------------------------------
    # Visual 5: Categorical Factor Impact (Work-Life Balance & OverTime vs Attrition)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # OverTime impact
    ot_df = df.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack() * 100
    ot_df.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[0], width=0.5, edgecolor='#1b3b5f')
    axes[0].set_title("Attrition Rate by OverTime Status", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Works OverTime", fontsize=11)
    axes[0].set_ylabel("Percentage of Employees (%)", fontsize=11)
    axes[0].set_xticklabels(['No Overtime', 'Works Overtime'], rotation=0)
    axes[0].legend(['Retained (No)', 'Attrition (Yes)'], frameon=True)
    for p in axes[0].patches:
        height = p.get_height()
        if height > 5:
            axes[0].annotate(f"{height:.1f}%",
                             (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                             ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            
    # WorkLifeBalance impact
    wlb_df = df.groupby('WorkLifeBalance')['Attrition'].value_counts(normalize=True).unstack() * 100
    wlb_labels = {1: '1 - Bad', 2: '2 - Good', 3: '3 - Better', 4: '4 - Best'}
    wlb_df.index = [wlb_labels.get(i, str(i)) for i in wlb_df.index]
    wlb_df.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d9534f'], ax=axes[1], width=0.55, edgecolor='#1b3b5f')
    axes[1].set_title("Attrition Rate by Work-Life Balance Rating", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Perceived Work-Life Balance", fontsize=11)
    axes[1].set_ylabel("Percentage of Employees (%)", fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(['Retained (No)', 'Attrition (Yes)'], frameon=True)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 5:
            axes[1].annotate(f"{height:.1f}%",
                             (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                             ha='center', va='center', fontsize=10, color='white', fontweight='bold')
            
    fig.suptitle("Visual 5: Bivariate Impact of Organizational Factors on Attrition Rate", fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    v5_path = os.path.join(output_dir, "eda_05_worklife_overtime_attrition.png")
    fig.savefig(v5_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visual 5] Saved workplace factors chart to {v5_path}")

    # -------------------------------------------------------------
    # Visual 6: Continuous Bivariate Scatterplot (Total Working Years vs Monthly Income)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.scatterplot(
        data=df, x='TotalWorkingYears', y='MonthlyIncome', hue='Attrition',
        palette={'No': '#2b5c8f', 'Yes': '#d9534f'}, alpha=0.75, s=45, edgecolor='none', ax=ax
    )
    # Fit regression trend line for retained
    sns.regplot(data=df[df['Attrition'] == 'No'], x='TotalWorkingYears', y='MonthlyIncome',
                scatter=False, ax=ax, color='#2b5c8f', line_kws={'linestyle': '--', 'linewidth': 1.8, 'label': 'Trend (Retained)'})
    # Fit regression trend line for attrition
    sns.regplot(data=df[df['Attrition'] == 'Yes'], x='TotalWorkingYears', y='MonthlyIncome',
                scatter=False, ax=ax, color='#d9534f', line_kws={'linestyle': '-', 'linewidth': 1.8, 'label': 'Trend (Attrition)'})
    
    ax.set_title("Visual 6: Bivariate Scatterplot of Career Experience vs. Monthly Salary by Attrition", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Total Working Experience (Years)", fontsize=11)
    ax.set_ylabel("Monthly Compensation ($ USD)", fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    v6_path = os.path.join(output_dir, "eda_06_scatter_experience_income.png")
    fig.savefig(v6_path, dpi=300)
    plt.close(fig)
    print(f"[Visual 6] Saved experience vs income scatterplot to {v6_path}")

    # -------------------------------------------------------------
    # Visual 7: Multivariate Pairplot Matrix
    # -------------------------------------------------------------
    pair_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 'Attrition']
    pair_grid = sns.pairplot(
        df[pair_features], hue='Attrition', palette={'No': '#2b5c8f', 'Yes': '#d9534f'},
        diag_kind='kde', plot_kws={'alpha': 0.6, 's': 25}, corner=True
    )
    pair_grid.fig.suptitle("Visual 7: Multivariate Pairwise Feature Interaction Matrix", fontsize=14, fontweight='bold', y=1.02)
    v7_path = os.path.join(output_dir, "eda_07_multivariate_pairplot.png")
    pair_grid.savefig(v7_path, dpi=300, bbox_inches='tight')
    plt.close(pair_grid.fig)
    print(f"[Visual 7] Saved pairplot matrix to {v7_path}")

    return {
        'visual_1': v1_path,
        'visual_2': v2_path,
        'visual_3': v3_path,
        'visual_4': v4_path,
        'visual_5': v5_path,
        'visual_6': v6_path,
        'visual_7': v7_path
    }

def stage_7_interact(df: pd.DataFrame, output_dir: str):
    """
    Stage 7: Interact - Generate interactive HTML visualization using Plotly.
    """
    try:
        import plotly.express as px
        fig = px.scatter(
            df, x='TotalWorkingYears', y='MonthlyIncome', color='Attrition',
            color_discrete_map={'No': '#2b5c8f', 'Yes': '#d9534f'},
            size='Age', hover_data=['JobRole', 'Department', 'OverTime'],
            title="Interactive Stage 7: Career Experience vs Salary with Multi-Attribute Hover"
        )
        fig.update_layout(template='plotly_white')
        html_path = os.path.join(output_dir, "eda_interactive_dashboard.html")
        fig.write_html(html_path)
        print(f"[Stage 7: INTERACT] Exported interactive visual to {html_path}")
        return html_path
    except ImportError:
        print("[Stage 7: INTERACT] Plotly not installed, skipping interactive HTML export.")
        return None

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'employee_attrition.csv')
    figures_path = os.path.join(os.path.dirname(__file__), '..', 'figures')
    df_raw = stage_1_acquire(data_path)
    stage_2_parse(df_raw)
    df_filtered = stage_3_filter(df_raw)
    stage_4_mine(df_filtered)
    stage_5_and_6_represent_and_refine(df_filtered, figures_path)
    stage_7_interact(df_filtered, figures_path)
