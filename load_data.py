import pandas as pd

electricity = pd.read_csv("electricity.csv")

print("Shape:", electricity.shape)
print(electricity.head())
print(electricity.info())
