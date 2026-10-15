# IBM HR Analytics Employee Attrition & Performance Dataset Dictionary

## Overview
- **Dataset Name:** IBM HR Analytics Employee Attrition & Performance
- **Source:** IBM Watson Analytics / Kaggle Open Datasets
- **Records (Rows):** 1,470 employees
- **Total Features (Columns):** 35 attributes
- **Target Variable:** `Attrition` (Binary: 'Yes' / 'No')
- **Objective:** Uncover key organizational drivers of employee turnover, explore demographic and workplace dynamics, and predict attrition probability using Deep Learning.

---

## Feature Schema and Descriptions

| # | Feature Name | Data Type | Domain / Values | Description |
|---|--------------|-----------|-----------------|-------------|
| 1 | `Age` | Numerical (Integer) | 18 – 60 | Age of the employee in years. |
| 2 | `Attrition` | Categorical (Binary) | 'Yes', 'No' | **Target variable**: whether the employee left the company. |
| 3 | `BusinessTravel` | Categorical (Nominal) | 'Travel_Rarely', 'Travel_Frequently', 'Non-Travel' | Frequency of business-related travel. |
| 4 | `DailyRate` | Numerical (Integer) | 102 – 1499 | Daily compensation rate in USD. |
| 5 | `Department` | Categorical (Nominal) | 'Sales', 'Research & Development', 'Human Resources' | Department within the organization. |
| 6 | `DistanceFromHome` | Numerical (Integer) | 1 – 29 | Commute distance from home to workplace (in miles). |
| 7 | `Education` | Numerical (Ordinal) | 1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor' | Formal education level achieved. |
| 8 | `EducationField` | Categorical (Nominal) | 'Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources' | Field of study/degree. |
| 9 | `EmployeeCount` | Numerical (Constant) | 1 | Constant identifier (always 1; candidate for removal). |
| 10 | `EmployeeNumber` | Numerical (Identifier) | 1 – 2068 | Unique employee identification number. |
| 11 | `EnvironmentSatisfaction` | Numerical (Ordinal) | 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High' | Satisfaction with physical/cultural work environment. |
| 12 | `Gender` | Categorical (Binary) | 'Male', 'Female' | Gender identity of the employee. |
| 13 | `HourlyRate` | Numerical (Integer) | 30 – 100 | Hourly wage rate in USD. |
| 14 | `JobInvolvement` | Numerical (Ordinal) | 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High' | Degree of engagement and psychological identification with work. |
| 15 | `JobLevel` | Numerical (Ordinal) | 1 – 5 | Hierarchical job level/band within the company. |
| 16 | `JobRole` | Categorical (Nominal) | 9 distinct roles (Sales Executive, Research Scientist, Lab Tech, etc.) | Specific role title held by the employee. |
| 17 | `JobSatisfaction` | Numerical (Ordinal) | 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High' | Subjective satisfaction with current job responsibilities. |
| 18 | `MaritalStatus` | Categorical (Nominal) | 'Single', 'Married', 'Divorced' | Marital status. |
| 19 | `MonthlyIncome` | Numerical (Continuous) | $1,009 – $19,999 | Monthly base salary in USD. |
| 20 | `MonthlyRate` | Numerical (Integer) | 2,094 – 26,999 | Monthly billing rate. |
| 21 | `NumCompaniesWorked` | Numerical (Integer) | 0 – 9 | Number of prior companies where the employee was employed. |
| 22 | `Over18` | Categorical (Constant) | 'Y' | Legal adulthood indicator (all 'Y'; candidate for removal). |
| 23 | `OverTime` | Categorical (Binary) | 'Yes', 'No' | Whether the employee works overtime hours. |
| 24 | `PercentSalaryHike` | Numerical (Integer) | 11% – 25% | Percentage increase in salary during the last annual review. |
| 25 | `PerformanceRating` | Numerical (Ordinal) | 3: 'Excellent', 4: 'Outstanding' | Recent formal performance review rating. |
| 26 | `RelationshipSatisfaction` | Numerical (Ordinal) | 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High' | Satisfaction with interpersonal relationships at work. |
| 27 | `StandardHours` | Numerical (Constant) | 80 | Standard bi-weekly working hours (all 80; candidate for removal). |
| 28 | `StockOptionLevel` | Numerical (Ordinal) | 0, 1, 2, 3 | Equity compensation grant level. |
| 29 | `TotalWorkingYears` | Numerical (Integer) | 0 – 40 | Total career working experience in years. |
| 30 | `TrainingTimesLastYear` | Numerical (Integer) | 0 – 6 | Number of professional development training sessions attended. |
| 31 | `WorkLifeBalance` | Numerical (Ordinal) | 1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best' | Perceived equilibrium between career demands and personal life. |
| 32 | `YearsAtCompany` | Numerical (Integer) | 0 – 40 | Number of continuous years tenure at IBM. |
| 33 | `YearsInCurrentRole` | Numerical (Integer) | 0 – 18 | Years spent in the current job role. |
| 34 | `YearsSinceLastPromotion`| Numerical (Integer) | 0 – 15 | Number of years elapsed since the most recent promotion. |
| 35 | `YearsWithCurrManager` | Numerical (Integer) | 0 – 17 | Number of years reporting to the current manager. |
