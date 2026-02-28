import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# ======================================================
# 1️⃣ Energy Trend
# ======================================================

def plot_energy_trend(df, building):

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df[building], linewidth=0.8)
    plt.title(f"Energy Consumption Over Time - {building}")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.tight_layout()
    plt.savefig("results/energy_trend.png")
    plt.close()


# ======================================================
# 2️⃣ Daily Average
# ======================================================

def plot_daily_average(df, building):

    os.makedirs("results", exist_ok=True)

    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.date
    daily_avg = df_copy.groupby("date")[building].mean()

    plt.figure(figsize=(14, 6))
    plt.plot(daily_avg.index, daily_avg.values)
    plt.title(f"Daily Average Energy - {building}")
    plt.xlabel("Date")
    plt.ylabel("Average Energy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/daily_average.png")
    plt.close()


# ======================================================
# 3️⃣ Anomaly Scatter Plot
# ======================================================

def plot_anomalies(df, building):

    if "anomaly" not in df.columns:
        return

    os.makedirs("results", exist_ok=True)

    normal = df[df["anomaly"] == 1]
    anomalies = df[df["anomaly"] == -1]

    plt.figure(figsize=(14, 6))
    plt.plot(normal["timestamp"], normal[building], alpha=0.7)
    plt.scatter(anomalies["timestamp"], anomalies[building])

    plt.title(f"Anomaly Detection - {building}")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.tight_layout()
    plt.savefig("results/anomaly_plot.png")
    plt.close()


# ======================================================
# 4️⃣ Anomaly Distribution (Pie)
# ======================================================

def plot_anomaly_distribution(df):

    if "anomaly" not in df.columns:
        return

    os.makedirs("results", exist_ok=True)

    normal = len(df[df["anomaly"] == 1])
    anomaly = len(df[df["anomaly"] == -1])

    plt.figure(figsize=(6, 6))
    plt.pie([normal, anomaly],
            labels=["Normal", "Anomaly"],
            autopct="%1.1f%%")
    plt.title("Anomaly Distribution")
    plt.tight_layout()
    plt.savefig("results/anomaly_distribution.png")
    plt.close()


# ======================================================
# 5️⃣ Monthly Anomaly Trend
# ======================================================

def plot_monthly_anomalies(df):

    if "anomaly" not in df.columns:
        return

    os.makedirs("results", exist_ok=True)

    anomaly_df = df[df["anomaly"] == -1].copy()
    anomaly_df["month"] = anomaly_df["timestamp"].dt.month
    monthly = anomaly_df.groupby("month").size()

    plt.figure(figsize=(8, 5))
    plt.bar(monthly.index, monthly.values)
    plt.xlabel("Month")
    plt.ylabel("Number of Anomalies")
    plt.title("Monthly Anomaly Trend")
    plt.tight_layout()
    plt.savefig("results/monthly_anomalies.png")
    plt.close()


# ======================================================
# 6️⃣ Hourly Heatmap
# ======================================================

def plot_hourly_heatmap(df, building):

    os.makedirs("results", exist_ok=True)

    df_copy = df.copy()
    df_copy["hour"] = df_copy["timestamp"].dt.hour
    df_copy["day"] = df_copy["timestamp"].dt.dayofweek

    pivot = df_copy.pivot_table(
        values=building,
        index="day",
        columns="hour",
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="coolwarm")
    plt.title("Average Energy Consumption Heatmap")
    plt.tight_layout()
    plt.savefig("results/hourly_heatmap.png")
    plt.close()


# ======================================================
# 7️⃣ Anomaly Type Distribution
# ======================================================

def plot_anomaly_types(df):

    if "type" not in df.columns:
        return

    os.makedirs("results", exist_ok=True)

    counts = df[df["anomaly"] == -1]["type"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Anomaly Type Distribution")
    plt.tight_layout()
    plt.savefig("results/anomaly_types.png")
    plt.close()


# ======================================================
# 8️⃣ Monthly Cost Impact
# ======================================================

def plot_cost_impact(df, building):

    if "anomaly" not in df.columns:
        return

    os.makedirs("results", exist_ok=True)

    anomaly_df = df[df["anomaly"] == -1].copy()
    avg_kwh_cost = 0.12

    anomaly_df["cost"] = anomaly_df[building] * avg_kwh_cost

    monthly_cost = anomaly_df.groupby(
        anomaly_df["timestamp"].dt.month
    )["cost"].sum()

    plt.figure(figsize=(8, 5))
    plt.bar(monthly_cost.index, monthly_cost.values)
    plt.xlabel("Month")
    plt.ylabel("Estimated Cost ($)")
    plt.title("Monthly Estimated Anomaly Cost")
    plt.tight_layout()
    plt.savefig("results/monthly_cost.png")
    plt.close()