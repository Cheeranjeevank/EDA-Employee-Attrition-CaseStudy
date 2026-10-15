"""
Generates a standalone interactive HTML visualization dashboard for Stage 7 (Interact)
Author: Cheeranjeevan K
Course: U21ADP05 - Exploratory Data Analysis and Visualization
"""

import os
import json
import pandas as pd

def generate_interactive_dashboard(data_path: str, output_path: str):
    df = pd.read_csv(data_path)
    
    # Sample down to 300 points for ultra-fast browser rendering and responsiveness
    sample_df = df[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
                    'Department', 'JobRole', 'OverTime', 'Attrition']].dropna()
    
    records = sample_df.to_dict(orient='records')
    records_json = json.dumps(records)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 7: Interactive EDA Dashboard - Employee Attrition Case Study</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #2b5c8f;
            --accent: #d9534f;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a, #2b5c8f);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; }}
        .header p {{ font-size: 14px; opacity: 0.9; }}
        .badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            background: var(--card-bg);
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
            align-items: center;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .control-group label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        select, input {{
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 14px;
            color: var(--text-main);
            background: #fff;
            min-width: 180px;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            padding: 18px;
            border-radius: 10px;
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
        }}
        .kpi-card.accent {{ border-left-color: var(--accent); }}
        .kpi-title {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; color: var(--text-main); }}
        .chart-container {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            height: 520px;
            position: relative;
        }}
        .chart-title {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Ben Fry Stage 7: Interactive Exploratory Visualization</h1>
        <p>Case Study: IBM HR Analytics Employee Attrition & Performance | Course: U21ADP05</p>
        <div class="badge">Dynamic Filterable Interface</div>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="deptSelect">Filter by Department</label>
            <select id="deptSelect" onchange="updateDashboard()">
                <option value="All">All Departments</option>
                <option value="Research & Development">Research & Development</option>
                <option value="Sales">Sales</option>
                <option value="Human Resources">Human Resources</option>
            </select>
        </div>
        <div class="control-group">
            <label for="otSelect">OverTime Filter</label>
            <select id="otSelect" onchange="updateDashboard()">
                <option value="All">All Employees</option>
                <option value="Yes">OverTime Only</option>
                <option value="No">No OverTime</option>
            </select>
        </div>
        <div class="control-group">
            <label for="attritionSelect">Attrition Status</label>
            <select id="attritionSelect" onchange="updateDashboard()">
                <option value="All">All Cohorts</option>
                <option value="Yes">Departed (Attrition = Yes)</option>
                <option value="No">Retained (Attrition = No)</option>
            </select>
        </div>
    </div>

    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-title">Filtered Employee Count</div>
            <div class="kpi-value" id="kpiCount">1,470</div>
        </div>
        <div class="kpi-card accent">
            <div class="kpi-title">Attrition Rate in Cohort</div>
            <div class="kpi-value" id="kpiRate">16.1%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Median Monthly Salary</div>
            <div class="kpi-value" id="kpiSalary">$4,919</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg. Total Working Experience</div>
            <div class="kpi-value" id="kpiExp">11.3 Yrs</div>
        </div>
    </div>

    <div class="chart-container">
        <div class="chart-title">Interactive Scatter Plot: Working Experience (Years) vs. Monthly Income ($ USD)</div>
        <canvas id="scatterChart"></canvas>
    </div>

    <script>
        const rawData = {records_json};
        let chartInstance = null;

        function updateDashboard() {{
            const dept = document.getElementById('deptSelect').value;
            const ot = document.getElementById('otSelect').value;
            const att = document.getElementById('attritionSelect').value;

            const filtered = rawData.filter(d => {{
                if (dept !== 'All' && d.Department !== dept) return false;
                if (ot !== 'All' && d.OverTime !== ot) return false;
                if (att !== 'All' && d.Attrition !== att) return false;
                return true;
            }});

            // Update KPIs
            document.getElementById('kpiCount').textContent = filtered.length.toLocaleString();
            const attrCount = filtered.filter(d => d.Attrition === 'Yes').length;
            const rate = filtered.length > 0 ? ((attrCount / filtered.length) * 100).toFixed(1) + '%' : '0.0%';
            document.getElementById('kpiRate').textContent = rate;
            
            const salaries = filtered.map(d => d.MonthlyIncome).sort((a,b) => a - b);
            const medSalary = salaries.length > 0 ? '$' + salaries[Math.floor(salaries.length / 2)].toLocaleString() : '$0';
            document.getElementById('kpiSalary').textContent = medSalary;
            
            const avgExp = filtered.length > 0 ? (filtered.reduce((acc, d) => acc + d.TotalWorkingYears, 0) / filtered.length).toFixed(1) + ' Yrs' : '0 Yrs';
            document.getElementById('kpiExp').textContent = avgExp;

            // Separate into Retained vs Attrition for Scatter Plot
            const retainedData = filtered.filter(d => d.Attrition === 'No').map(d => ({{
                x: d.TotalWorkingYears,
                y: d.MonthlyIncome,
                role: d.JobRole,
                dept: d.Department,
                ot: d.OverTime
            }}));

            const attritionData = filtered.filter(d => d.Attrition === 'Yes').map(d => ({{
                x: d.TotalWorkingYears,
                y: d.MonthlyIncome,
                role: d.JobRole,
                dept: d.Department,
                ot: d.OverTime
            }}));

            if (chartInstance) {{
                chartInstance.data.datasets[0].data = retainedData;
                chartInstance.data.datasets[1].data = attritionData;
                chartInstance.update();
            }} else {{
                const ctx = document.getElementById('scatterChart').getContext('2d');
                chartInstance = new Chart(ctx, {{
                    type: 'scatter',
                    data: {{
                        datasets: [
                            {{
                                label: 'Retained (Attrition = No)',
                                data: retainedData,
                                backgroundColor: 'rgba(43, 92, 143, 0.7)',
                                borderColor: '#1b3b5f',
                                pointRadius: 4.5,
                                pointHoverRadius: 7
                            }},
                            {{
                                label: 'Departed (Attrition = Yes)',
                                data: attritionData,
                                backgroundColor: 'rgba(217, 83, 79, 0.85)',
                                borderColor: '#a94442',
                                pointRadius: 5.5,
                                pointHoverRadius: 8
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const p = context.raw;
                                        return [
                                            context.dataset.label,
                                            'Experience: ' + p.x + ' Years',
                                            'Monthly Salary: $' + p.y.toLocaleString(),
                                            'Role: ' + p.role,
                                            'OverTime: ' + p.ot
                                        ];
                                    }}
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{ display: true, text: 'Total Working Experience (Years)', font: {{ size: 13, weight: 'bold' }} }},
                                min: 0
                            }},
                            y: {{
                                title: {{ display: true, text: 'Monthly Income ($ USD)', font: {{ size: 13, weight: 'bold' }} }},
                                min: 0
                            }}
                        }}
                    }}
                }});
            }}
        }}

        window.onload = updateDashboard;
    </script>
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated Stage 7 Interactive Dashboard at {output_path}")

if __name__ == '__main__':
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'employee_attrition.csv')
    html_file = os.path.join(os.path.dirname(__file__), '..', 'figures', 'eda_interactive_dashboard.html')
    generate_interactive_dashboard(data_file, html_file)
