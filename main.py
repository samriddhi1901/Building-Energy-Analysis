from src.load_data import load_data
from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.business_insights import generate_business_insights
from src.modeling import run_model
from src.visualization import (
    plot_energy_trend,
    plot_daily_average,
    plot_anomalies,
    plot_anomaly_distribution,
    plot_monthly_anomalies,
    plot_hourly_heatmap,
    plot_anomaly_types,
    plot_cost_impact
)

def main():

    # Step 1: Load Data
    df = load_data("data/electricity.csv")

    # Step 2: Clean Data
    df = clean_data(df)
    print("After Cleaning Shape:", df.shape)
    print(df.head())

    # Step 3: Select a building column
    building = df.columns[5]
    print("Selected Building:", building)

    # Step 4: Feature Engineering
    df = create_features(df, building)

    # Step 5: Run ML Model
    df = run_model(df, building)

    # Step 6: Visualization
    print("\n========== VISUALIZATION ==========")
    plot_anomaly_distribution(df)
    plot_monthly_anomalies(df)
    plot_hourly_heatmap(df, building)
    plot_anomaly_types(df)
    plot_cost_impact(df, building)

    # Step 7: Business Insights  ✅ MOVED INSIDE
    generate_business_insights(df, building)

    print("\nPipeline Completed Successfully ✅")


if __name__ == "__main__":
    main()