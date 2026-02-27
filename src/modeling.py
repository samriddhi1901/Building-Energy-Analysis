#isolation forest model imports
from sklearn.ensemble import IsolationForest

# Local Outlier model imports
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

#Robust Covarience model imports
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

#==================================
#Isolationforest model function
#==================================
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

#====================================
#Local Outliers model's function
#====================================
def run_lof_model(df, building):

    print("\n========== MODELING (LOF) ==========")

    if building not in df.columns:
        print("Building column not found.")
        return df

    data = df[['timestamp', building]].dropna()

    print("Data Shape Before Model:", data.shape)

    if data.shape[0] == 0:
        print("No data available for modeling.")
        return df

    X = data[[building]]

    # Scale data (important for LOF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01
    )

    y_pred = model.fit_predict(X_scaled)

    data['anomaly'] = (y_pred == -1).astype(int)

    anomaly_count = data['anomaly'].sum()

    print("Total Anomalies Detected:", anomaly_count)
    print("Anomaly Percentage:",
          round(anomaly_count / len(data) * 100, 2), "%")

    file_name="Lof_anomaly_results.csv" #The file name when saved
    data.to_csv(file_name, index=False)
    print("Results saved as {}".format(file_name))

    print("====================================")

    return data

#==================================
#Robust covarience model Function
#==================================
def run_robust_model(df, building):

    print("\n========== MODELING (Robust Covariance) ==========")

    if building not in df.columns:
        print("Building column not found.")
        return df

    data = df[['timestamp', building]].dropna()

    print("Data Shape Before Model:", data.shape)

    if data.shape[0] == 0:
        print("No data available for modeling.")
        return df

    X = data[[building]]

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = EllipticEnvelope(
        contamination=0.01,
        random_state=42
    )

    model.fit(X_scaled)

    y_pred = model.predict(X_scaled)

    data['anomaly'] = (y_pred == -1).astype(int)

    anomaly_count = data['anomaly'].sum()

    print("Total Anomalies Detected:", anomaly_count)
    print("Anomaly Percentage:",
          round(anomaly_count / len(data) * 100, 2), "%")
    
    file_name="ml_anomaly_results.csv" #saves file name in variable
    
    data.to_csv(file_name, index=False)
    print("Results saved as {}".format(file_name))

    print("===============================================")

    return data
