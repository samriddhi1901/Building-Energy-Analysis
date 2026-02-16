# Building Energy Consumption Analysis

## 📌 Project Overview
This project analyzes large-scale building electricity consumption data using time-series analysis techniques.  
The dataset is taken from the **Building Data Genome Project 2**, which contains hourly energy readings of more than 1500 buildings over two years.

The goal of the project is to understand energy usage behavior, detect unusual consumption patterns, and prepare data for future machine learning prediction models.

---

## 📂 Dataset Information
- Dataset: Building Data Genome Project 2
- Time Period: 2016 – 2017
- Frequency: Hourly readings
- Buildings: 1500+ buildings
- Type: Multivariate Time-Series Data

Each column represents a building and each row represents energy consumption at a specific timestamp.

---

## 🔍 Analysis Performed
The following analysis was conducted:

### 1. Data Understanding
- Loaded large dataset efficiently
- Checked dataset dimensions and structure
- Identified building-wise readings

### 2. Missing Value Analysis
- Found incomplete sensor readings
- Total missing values: 1,312,095

### 3. Pattern Analysis
- Daily consumption trends
- Weekly usage behavior
- Work-hour vs non-work-hour patterns

### 4. Anomaly Detection
- Statistical threshold: Mean + 3 × Standard Deviation
- Detected abnormal consumption spikes
- Total anomalies found: 9

---

## 🛠 Technologies Used
- Python
- Pandas
- Matplotlib
- VS Code
- Git & GitHub

---

## 📊 Project Structure

