#missing values check
import pandas as pd

df = pd.read_csv("electricity.csv")

missing = df.isnull().sum().sum()
print("Total missing values:", missing)

#Daily consumption pattern
import matplotlib.pyplot as plt

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

building = df.columns[5]

hourly = df.groupby('hour')[building].mean()

plt.plot(hourly)
plt.title("Average Hourly Consumption Pattern")
plt.xlabel("Hour of Day")
plt.ylabel("Energy Usage")
plt.show()

#Weekly Pattern
df['day'] = df['timestamp'].dt.dayofweek

weekly = df.groupby('day')[building].mean()

plt.plot(weekly)
plt.title("Weekly Energy Pattern (0=Mon)")
plt.xlabel("Day")
plt.ylabel("Energy Usage")
plt.show()

#detect possible anomalies
import numpy as np

series = df[building]
threshold = series.mean() + 3*series.std()

anomalies = series[series > threshold]

print("Number of anomalies:", len(anomalies))
