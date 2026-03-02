# Building-energy-anomaly-detection-ml

## 📌 Project Overview

This project performs large-scale time-series analysis and machine learning–based anomaly detection on commercial building electricity consumption data.

The dataset is taken from the Building Data Genome Project 2, which contains hourly energy readings of more than 1500 buildings across two years (2016–2017).

The primary objective is to:

Understand building-wise energy usage behavior

Handle large-scale missing data

Engineer meaningful time-based features

Detect abnormal energy consumption patterns using unsupervised machine learning

This project follows a complete ML pipeline:

Data Loading → Data Cleaning → Feature Engineering → Anomaly Detection

## 📂 Dataset Information

Dataset: Building Data Genome Project 2

Time Period: 2016 – 2017

Frequency: Hourly readings

Total Rows: 17,544 timestamps

Total Columns: 1,579 (1 timestamp + 1,578 buildings)

Type: Multivariate Time-Series Data

Each row represents a timestamp and each column represents a building’s electricity consumption.

## 🔍 Data Preprocessing & Analysis
### 1️⃣ Data Understanding

Loaded large dataset efficiently

Converted timestamp to datetime format

Verified dataset dimensions and structure

### 2️⃣ Missing Value Handling

Total missing values identified: 1,312,095

Applied forward fill method to maintain time continuity

Preserved all rows after cleaning

### 3️⃣ Feature Engineering

Generated time-based and statistical features:

Hour of day

Day of week

Weekend indicator

Lag features (1-hour and 24-hour)

24-hour rolling mean

24-hour rolling standard deviation

These features help the model learn temporal consumption patterns.

## 🤖 Machine Learning – Anomaly Detection

Unsupervised machine learning techniques were applied to detect abnormal electricity consumption patterns in commercial buildings.

The system uses three anomaly detection models to improve reliability.

### ✅ 1️⃣ Isolation Forest

Contamination: 1%

Learns normal energy usage behavior

Detects rare global deviations in consumption

#### 📊 Results

Total Anomalies Detected: 175

Anomaly Percentage: 1%

Results saved as: ml_anomaly_results.csv

Isolation Forest isolates rare observations by randomly partitioning the data. Shorter path length = higher anomaly probability.

### ✅ 2️⃣ Local Outlier Factor (LOF)

n_neighbors: 20

Contamination: 1%

Data scaled using StandardScaler

LOF compares each data point to its nearest neighbors and identifies points that have significantly lower local density.

#### 📊 Results

Detects anomalies based on neighborhood density

Results saved as: Lof_anomaly_results.csv

Useful for detecting local irregularities in energy consumption patterns.

### ✅ 3️⃣ Robust Covariance (Elliptic Envelope)

Contamination: 1%

Uses Mahalanobis distance

Data scaled using StandardScaler

This model assumes normal energy consumption follows a Gaussian distribution and creates an elliptical boundary around normal data points.

#### 📊 Results

Detects anomalies outside the statistical boundary

Results saved as: ml_anomaly_results.csv

Called "robust" because it reduces the influence of extreme values during covariance estimation.

## 🚀 Why Multiple Models?

Since the dataset does not contain labeled anomalies, all models are unsupervised.

Each model detects different types of anomalies:

Isolation Forest → Global anomalies

LOF → Local density anomalies

Robust Covariance → Statistical distribution anomalies

Using multiple approaches increases reliability in detecting unusual energy behavior.

# Technologies Used

Python

Pandas

NumPy

Scikit-learn

VS Code

Git & GitHub

# 📁 Project Structure
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
