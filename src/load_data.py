import pandas as pd
import os

def load_data(filepath):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, filepath)

    print("Trying to load from:", full_path)

    df = pd.read_csv(full_path, na_values=['null'])  # keep 'null' in quotes
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("Original Dataset Shape:", df.shape)

    return df
