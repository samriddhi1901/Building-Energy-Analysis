def clean_data(df):

    print("\n========== DATA CLEANING ==========")

    initial_rows = df.shape[0]
    print("Initial Rows:", initial_rows)

    df = df.drop_duplicates()

    # Count missing
    total_missing = df.isnull().sum().sum()
    print("Total Missing Values Before:", total_missing)

    # Only forward fill
    df = df.fillna(method='ffill')

    # DO NOT drop entire rows blindly
    # Instead drop rows only if timestamp is missing
    df = df.dropna(subset=['timestamp'])

    final_rows = df.shape[0]

    print("Rows After Cleaning:", final_rows)
    print("Rows Removed:", initial_rows - final_rows)

    print("===================================")

    return df
