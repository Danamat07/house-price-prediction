import pandas as pd

def load_data(train_path="../data/train.csv", test_path="../data/test.csv"):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Coloane train: {train_df.shape[1]}")
    print(f"Coloane test: {test_df.shape[1]}")

    return train_df, test_df
