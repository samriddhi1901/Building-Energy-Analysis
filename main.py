from src.load_data import load_data
from src.data_cleaning import clean_data
from src.feature_engineering import create_features
from src.modeling import run_model   # make sure modeling.py exists

def main():

    # Step 1: Load Data
    df = load_data("data/electricity.csv")

    # Step 2: Clean Data
    df = clean_data(df)
    print("After Cleaning Shape:", df.shape)
    print(df.head())


    # Step 3: Select a building column
    # First column is timestamp, so start from index 1
    building = df.columns[5]   # You can change index if needed
    print("Selected Building:", building)

    # Step 4: Feature Engineering
    df = create_features(df, building)

    # Step 5: Run ML Model
    df = run_model(df, building)

    print("\nPipeline Completed Successfully ✅")

if __name__ == "__main__":
    main()
