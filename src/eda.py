import matplotlib.pyplot as plt

def plot_time_series(df, building):
    plt.figure(figsize=(12,5))
    plt.plot(df['timestamp'], df[building])
    plt.title(f"Energy Usage Pattern: {building}")
    plt.xlabel("Time")
    plt.ylabel("Energy Consumption")
    plt.show()
