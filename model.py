import numpy as np
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load dataset from data directory
df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["target"]).values
y = df["target"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Model definition
model = LogisticRegression(max_iter=200)

# Training
model.fit(X_train, y_train)

# Evaluation
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Validation accuracy: {acc:.4f}")