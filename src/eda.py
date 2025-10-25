import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 200)

def quick_eda(path="../data/train.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nInfo (first 10 lines):")
    print(df.info())
    print("\nSalePrice description:")
    print(df['SalePrice'].describe())

    # Target distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice distribution")
    plt.savefig("../models/eda_saleprice_dist.png")
    print("Saved ../models/eda_saleprice_dist.png")

    # Log transform distribution
    plt.figure(figsize=(8,5))
    sns.histplot(np.log1p(df['SalePrice']), kde=True)
    plt.title("log1p(SalePrice) distribution")
    plt.savefig("../models/eda_saleprice_log_dist.png")
    print("Saved ../models/eda_saleprice_log_dist.png")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nTop missing features:")
    print(missing.head(50))

    # Correlations with SalePrice (top 20 numeric)
    corr = df.select_dtypes(include=['int64','float64']).corr()['SalePrice'].sort_values(ascending=False)
    print("\nTop correlations with SalePrice:")
    print(corr.head(20))

    # Scatter for top 3 numeric features (if exist)
    top_feats = corr.index[1:4].tolist()
    for feat in top_feats:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[feat], y=df['SalePrice'])
        plt.title(f"SalePrice vs {feat}")
        fname = f"../models/eda_scatter_{feat}.png"
        plt.savefig(fname)
        print("Saved", fname)

if __name__ == "__main__":
    quick_eda()
