# UniteUs Log Generator

This Python script generates realistic dummy log data for the UniteUs app across different log sources: DataDog, Analytics, and Backend. The generated logs are saved in CSV format for later analysis.

## Features

- Generate synthetic log data that resembles real logs from:
  - DataDog monitoring
  - User analytics
  - Backend services
- Configurable settings:
  - Time period (number of days)
  - Log volume (low, medium, high, very_high)
  - Error rate
  - Random seed for reproducibility
- Realistic log formats with appropriate fields for each source
- Export to CSV files for analysis
- Rich metadata relevant to the UniteUs application domain

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - faker

You can install required packages using:

```bash
pip install pandas faker
# log-gen
