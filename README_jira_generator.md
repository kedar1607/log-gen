# UniteUs Jira Issue Generator

This tool generates realistic Jira issues based on actual log data from the UniteUs application. It's designed specifically for creating data that can be used to train and validate machine learning models for Root Cause Analysis (RCA).

## Features

- Generates Jira issues from real application logs (DataDog, Backend)
- Creates a specified distribution of issues with and without RCA (default: 80% with RCA, 20% without)
- Automatically creates train/test splits for machine learning
- Includes detailed issue descriptions, priorities, statuses, and components based on log data
- Generates realistic root cause analysis for resolved issues

## Prerequisites

- Python 3.6+
- Required Python packages:
  - pandas
  - faker

## Usage

### Basic Usage

```bash
python jira_issue_generator.py
```

This will:
1. Load existing log data from the `logs/` directory
2. Generate 100 Jira issues (default) with 80% having RCA information
3. Save the results to `jira_issues/jira_issues.csv`
4. Create train/test splits for machine learning (80/20 split)

### Command Line Options

```
python jira_issue_generator.py --help
```

Available options:

- `--logs-dir`: Directory containing log CSV files (default: `logs/`)
- `--output-dir`: Directory to save generated Jira issues (default: `jira_issues/`)
- `--rca-ratio`: Proportion of issues that should have RCA information (default: 0.8)
- `--count`: Number of Jira issues to generate (default: 100)
- `--test-split`: Proportion of data to use for testing (default: 0.2)

### Examples

Generate 200 issues with 70% RCA coverage:
```bash
python jira_issue_generator.py --count 200 --rca-ratio 0.7
```

Change the test set size to 30%:
```bash
python jira_issue_generator.py --test-split 0.3
```

## Output Files

The script generates the following files in the output directory:

- `jira_issues.csv`: All generated Jira issues
- `train_issues.csv`: Training dataset (80% by default)
- `test_issues.csv`: Testing dataset (20% by default)

## CSV Format

The generated CSV files include the following fields:

- `key`: Jira issue key (e.g., "UNITE-1234")
- `summary`: Issue summary/title
- `description`: Detailed issue description
- `issue_type`: Bug, Incident, Task, etc.
- `priority`: Blocker, Critical, Major, Minor, or Trivial
- `status`: Open, In Progress, Code Review, Testing, Done, or Closed
- `assignee`: Issue assignee name
- `reporter`: Issue reporter name
- `components`: Affected components/services
- `created_at`: Issue creation timestamp
- `resolved_at`: Issue resolution timestamp (if resolved)
- `affected_services`: List of affected services
- `environment`: Environment where the issue occurred (production, staging, etc.)
- `has_rca`: Boolean indicating if RCA is available (true/false)

For issues with RCA, additional fields are included:

- `rca_root_cause_category`: Category of the root cause
- `rca_affected_components`: Components affected by the issue
- `rca_resolution_steps`: Steps taken to resolve the issue
- `rca_prevention_steps`: Steps to prevent similar issues
- `rca_description`: Detailed description of the root cause

## Integration with log_generator.py

This script is designed to work with the output from `log_generator.py`. To generate both logs and Jira issues:

1. Generate logs:
   ```bash
   python log_generator.py
   ```

2. Generate Jira issues based on those logs:
   ```bash
   python jira_issue_generator.py
   ```

## Machine Learning Applications

The generated data is suitable for various machine learning tasks:

- **Classification**: Predict if an issue has a known root cause
- **Multi-class classification**: Predict the root cause category
- **Named Entity Recognition**: Extract affected components from descriptions
- **Clustering**: Group similar issues by root cause patterns

## Future Enhancements

- Add support for generating custom fields
- Implement more sophisticated issue relationship modeling (parent/child, duplicates)
- Add time-based trends in issue resolution