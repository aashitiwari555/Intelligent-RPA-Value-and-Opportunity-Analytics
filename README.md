# Intelligent RPA Value & Automation Opportunity Analytics

A Streamlit-based interactive analytics dashboard analysing **50,000 enterprise automation projects** to surface automation performance KPIs, business value drivers, and prioritised investment opportunities.

---

## Problem Statement

Organisations invest heavily in Robotic Process Automation (RPA), yet struggle to understand which automation types, industries, and project characteristics consistently deliver the highest return. This dashboard transforms raw automation project data into actionable intelligence — identifying high-performing automation strategies, underperforming investment areas, and opportunity priorities for future automation spend.

---

## Dataset Description

| Property | Value |
|---|---|
| File | `dataset/automation_projects.csv` |
| Rows | 50,000 enterprise automation projects |
| Columns | 18 |
| Date range | 2015–2025 |
| Industries | 26 |
| Departments | 30 |
| Countries | 56 |
| Automation types | 25 |
| Implementation partners | 38 |

### Key Columns

| Column | Description |
|---|---|
| `project_id` | Unique project identifier |
| `company_id` | Company identifier (multiple projects per company) |
| `project_name` | Descriptive project name |
| `start_date` | Project start date (YYYY-MM-DD) |
| `completion_date` | Project completion date; missing (~30%) for Active projects |
| `project_status` | Active / Completed / Planned |
| `automation_type` | Type of automation (25 categories) |
| `robots_deployed` | Number of software robots deployed (1–50) |
| `budget_usd` | Total project budget in USD |
| `annual_savings_usd` | Estimated annual cost savings in USD |
| `roi_percent` | ROI defined as `(annual_savings_usd / budget_usd) × 100` |
| `department` | Business department running the project |
| `implementation_partner` | Consulting/SI firm |
| `country` | Country of deployment |
| `industry` | Industry sector |
| `employee_hours_saved` | Annual employee hours saved |
| `ai_enabled` | Whether AI/ML was used (Yes/No) |
| `cloud_deployment` | Whether cloud infrastructure was used (Yes/No) |

> ⚠️ **ROI Definition:** `roi_percent = (annual_savings_usd / budget_usd) × 100`
> This is a **savings-to-investment ratio**, NOT conventional net-return ROI.
> All labels in the dashboard use "ROI (Savings/Budget %)" to make this explicit.

**Original dataset source:** [Kaggle — Enterprise RPA Automation Projects Dataset](https://www.kaggle.com/datasets/shikhariitb/enterprise-rpa-automation-projects-dataset)

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data manipulation and aggregation |
| NumPy | Numerical operations |
| Plotly | Interactive charts and visualisations |
| python-docx | Programmatic Word document generation |

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone or download the project
cd IBM_Internship_RPA_Project

# 2. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## Key Features & KPIs

### Executive Overview
| KPI | Value (full dataset) |
|---|---|
| Total Projects | 50,000 |
| Avg ROI (Savings/Budget %) | 174.6% |
| Total Budget Invested | ~$9.1 Billion |
| Total Annual Savings | ~$15.9 Billion |
| Total Employee Hours Saved | ~3.1 Billion |
| Completion Rate | 55.4% |
| AI Adoption Rate | 55.1% |
| Cloud Adoption Rate | 64.8% |

### Dashboard Tabs
1. **📊 Executive Overview** — KPI scorecards, project status donut, year-over-year adoption trend
2. **⚙️ Performance Analysis** — ROI by automation type, savings by industry, budget vs savings scatter, hours saved, robots deployed
3. **💡 Value Drivers** — AI vs Non-AI comparison, Cloud vs Non-Cloud, 2×2 heatmap, project duration analysis
4. **🎯 Opportunity Analysis** — Opportunity Score bubble maps, ranked scored tables, High Investment / Low Return flags

### Interactive Filters (Sidebar)
- Industry (26 options)
- Department (30 options)
- Automation Type (25 options)
- Country (56 options)
- Project Status (Active / Completed / Planned)
- AI Enabled (All / Yes / No)
- Cloud Deployment (All / Yes / No)

All filters apply globally across all four dashboard tabs.

---

## Project Structure

```
IBM_Internship_RPA_Project/
├── app.py                          # Main Streamlit dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── project_report.docx             # Written project report
├── rpa-analytics-plan.md           # Implementation plan
└── dataset/
    └── automation_projects.csv     # 50,000-row dataset
```

---

## Opportunity Score Methodology

Each automation type and industry is scored 0–100 using three equal-weight components:

| Component | Metric | Weight |
|---|---|---|
| ROI Score | Percentile rank of median ROI | 33% |
| Savings Score | Percentile rank of total annual savings | 33% |
| Productivity Score | Percentile rank of avg employee hours saved | 34% |

A **"High Investment / Low Return"** flag is raised when a group has above-median total budget **and** below-33rd-percentile median ROI. No machine learning is used — the framework is fully transparent and auditable.

---

## Notes

- Missing `completion_date` values (~30%) are expected: they correspond exclusively to **Active** projects. No imputation is applied.
- ROI values are capped at **500%** in the dataset. The dashboard uses **median ROI** (not mean) for group comparisons to reduce the effect of this cap.
- The scatter plots sample up to 5,000 points for browser performance; all aggregated KPIs use the full filtered dataset.
