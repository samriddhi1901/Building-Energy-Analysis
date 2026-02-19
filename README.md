⚡ Building Energy Consumption Anomaly Detection
📌 Project Overview

This project performs large-scale time-series analysis and machine learning–based anomaly detection on commercial building electricity consumption data.

The dataset is taken from the Building Data Genome Project 2, which contains hourly energy readings of more than 1500 buildings across two years (2016–2017).

The primary objective is to:

Understand building-wise energy usage behavior

Handle large-scale missing data

Engineer meaningful time-based features

Detect abnormal energy consumption patterns using unsupervised machine learning

This project follows a complete ML pipeline:

Data Loading → Data Cleaning → Feature Engineering → Anomaly Detection

📂 Dataset Information

Dataset: Building Data Genome Project 2

Time Period: 2016 – 2017

Frequency: Hourly readings

Total Rows: 17,544 timestamps

Total Columns: 1,579 (1 timestamp + 1,578 buildings)

Type: Multivariate Time-Series Data

Each row represents a timestamp and each column represents a building’s electricity consumption.

🔍 Data Preprocessing & Analysis
1️⃣ Data Understanding

Loaded large dataset efficiently

Converted timestamp to datetime format

Verified dataset dimensions and structure

2️⃣ Missing Value Handling

Total missing values identified: 1,312,095

Applied forward fill method to maintain time continuity

Preserved all rows after cleaning

3️⃣ Feature Engineering

Generated time-based and statistical features:

Hour of day

Day of week

Weekend indicator

Lag features (1-hour and 24-hour)

24-hour rolling mean

24-hour rolling standard deviation

These features help the model learn temporal consumption patterns.

🤖 Machine Learning – Anomaly Detection

Unsupervised learning techniques were applied to detect abnormal electricity consumption.

✅ Isolation Forest

Contamination: 1%

Learns normal energy usage behavior

Detects rare consumption deviations

📊 Results

Total Anomalies Detected: 175

Anomaly Percentage: 1%

Anomalies indicate unusual spikes or drops in electricity usage that may represent:

Equipment malfunction

Sensor irregularities

Operational inefficiencies

Abnormal building usage

The final results are saved as:

ml_anomaly_results.csv

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

VS Code

Git & GitHub

📁 Project Structure
Building-Energy-Analysis
│
├── data/
│   └── electricity.csv
│
├── src/
│   ├── load_data.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── modeling.py
│
├── main.py
├── ml_anomaly_results.csv
├── README.md

🚀 How to Run
