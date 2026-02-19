def statistical_anomaly_detection(df, building):

    series = df[building]
    threshold = series.mean() + 3 * series.std()

    df['stat_anomaly'] = (series > threshold).astype(int)

    anomaly_count = df['stat_anomaly'].sum()

    print("Statistical Anomalies:", anomaly_count)

    return df
