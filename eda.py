import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("electricity.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# choose one building
building = df.columns[10]

plt.figure(figsize=(12,5))
plt.plot(df['timestamp'], df[building])
plt.title(f"Energy Usage Pattern: {building}")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.show()
