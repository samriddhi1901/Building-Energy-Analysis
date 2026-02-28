import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_energy_trend(df, building):
    """
    Plots full time-series energy consumption for selected building
    """

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df[building], linewidth=0.8)
    plt.title(f"Energy Consumption Over Time - {building}")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.tight_layout()
    plt.savefig("results/energy_trend.png")
    plt.close()

    print("✅ Energy trend plot saved: results/energy_trend.png")


def plot_daily_average(df, building):
    """
    Plots daily average consumption
    """

    os.makedirs("results", exist_ok=True)

    df["date"] = df["timestamp"].dt.date
    daily_avg = df.groupby("date")[building].mean()

    plt.figure(figsize=(14, 6))
    plt.plot(daily_avg.index, daily_avg.values)
    plt.title(f"Daily Average Energy Consumption - {building}")
    plt.xlabel("Date")
    plt.ylabel("Average Energy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/daily_average.png")
    plt.close()

    print("✅ Daily average plot saved: results/daily_average.png")


def plot_anomalies(df, building):
    """
    Plots detected anomalies from Isolation Forest
    Assumes anomaly column contains:
    1  → Normal
    -1 → Anomaly
    """

    if "anomaly" not in df.columns:
        print("⚠ No anomaly column found. Run modeling first.")
        return

    os.makedirs("results", exist_ok=True)

    normal = df[df["anomaly"] == 1]
    anomalies = df[df["anomaly"] == -1]

    plt.figure(figsize=(14, 6))

    plt.plot(normal["timestamp"], normal[building], 
             label="Normal", alpha=0.7)

    plt.scatter(anomalies["timestamp"], anomalies[building], 
                label="Anomaly")

    plt.title(f"Anomaly Detection - {building}")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/anomaly_plot.png")
    plt.close()

    print("✅ Anomaly plot saved: results/anomaly_plot.png")