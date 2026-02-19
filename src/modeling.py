from sklearn.ensemble import IsolationForest

def run_model(df, building):

    print("\n========== MODELING ==========")

    if building not in df.columns:
        print("Building column not found.")
        return df

    data = df[['timestamp', building]].dropna()

    print("Data Shape Before Model:", data.shape)

    if data.shape[0] == 0:
        print("No data available for modeling.")
        return df

    X = data[[building]]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    model.fit(X)

    data['anomaly'] = model.predict(X)
    data['anomaly'] = data['anomaly'].map({-1: 1, 1: 0})

    anomaly_count = data['anomaly'].sum()

    print("Total Anomalies Detected:", anomaly_count)
    print("Anomaly Percentage:",
          round(anomaly_count / len(data) * 100, 2), "%")
    data.to_csv("ml_anomaly_results.csv", index=False)
    print("Results saved as ml_anomaly_results.csv")

    print("==============================")

    return data
