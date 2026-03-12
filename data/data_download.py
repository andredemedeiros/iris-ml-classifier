import pandas as pd

# Wine Quality — UCI (CC BY 4.0) | 6497 samples | 11 features | 7 classes
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")
df = df.rename(columns={"quality": "target"})

df.to_csv("data/wine.csv", index=False)
print(f"Saved: {len(df)} samples, {df.shape[1]-1} features, {df['target'].nunique()} classes")