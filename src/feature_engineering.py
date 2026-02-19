def create_features(df, building):

    print("\n========== FEATURE ENGINEERING ==========")

    if building not in df.columns:
        print("Building column not found.")
        return df

    # Create time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Create lag features
    df['lag_1'] = df[building].shift(1)
    df['lag_24'] = df[building].shift(24)

    # Rolling features
    df['rolling_mean_24'] = df[building].rolling(window=24).mean()
    df['rolling_std_24'] = df[building].rolling(window=24).std()

    # Only drop rows where selected building is NaN
    df = df.dropna(subset=[building])

    print("Shape After Feature Engineering:", df.shape)
    print("=========================================")

    return df
