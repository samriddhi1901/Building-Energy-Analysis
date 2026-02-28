import os
import pandas as pd
import numpy as np


def generate_business_insights(df, building):

    print("\n========== BUSINESS INSIGHTS ==========")

    os.makedirs("results", exist_ok=True)

    if "anomaly" not in df.columns:
        print("No anomaly column found. Run model first.")
        return

    anomaly_df = df[df["anomaly"] == -1].copy()

    if anomaly_df.empty:
        print("No anomalies detected.")
        return

    # -----------------------------
    # 1️⃣ Cost Estimation
    # -----------------------------
    avg_kwh_cost = 0.12  # Example electricity rate ($/kWh)
    total_anomaly_energy = anomaly_df[building].sum()
    estimated_cost = total_anomaly_energy * avg_kwh_cost

    print(f"\nEstimated cost impact from anomalies: ${estimated_cost:,.2f}")

    # -----------------------------
    # 2️⃣ Monthly Pattern
    # -----------------------------
    anomaly_df["month"] = anomaly_df["timestamp"].dt.month
    monthly_pattern = anomaly_df.groupby("month").size()

    print("\nAnomalies by Month:")
    print(monthly_pattern)

    # -----------------------------
    # 3️⃣ Peak Anomaly Hours
    # -----------------------------
    anomaly_df["hour"] = anomaly_df["timestamp"].dt.hour
    peak_hours = (
        anomaly_df.groupby("hour")
        .size()
        .sort_values(ascending=False)
    )

    print("\nTop 5 Peak Anomaly Hours:")
    print(peak_hours.head())

    # -----------------------------
    # 4️⃣ Anomaly Type Classification
    # -----------------------------
    mean_val = df[building].mean()
    std_val = df[building].std()

    anomaly_df["z_score"] = (
        anomaly_df[building] - mean_val
    ) / std_val

    anomaly_df["type"] = pd.cut(
        anomaly_df["z_score"],
        bins=[-np.inf, -2, 2, np.inf],
        labels=["Drop", "Normal", "Spike"]
    )

    print("\nAnomaly Type Distribution:")
    print(anomaly_df["type"].value_counts())

    # -----------------------------
    # Save Full Business Report
    # -----------------------------
    anomaly_df.to_csv(
        "results/anomaly_business_analysis.csv",
        index=False
    )

    print("\nBusiness insights saved in results/anomaly_business_analysis.csv")