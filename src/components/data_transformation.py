# initialize preprocessor
from src.utils import DataPreprocessor
import pandas as pd


pre = DataPreprocessor()

# preprocess train
train_df = pd.read_csv(r"artifacts\train.csv")
train_processed = pre.fit_transform(train_df)

# preprocess test (uses train mappings)
test_df = pd.read_csv(r"artifacts\test.csv")
test_processed = pre.transform(test_df)

# split
X_train = train_processed.drop("is_attributed", axis=1)
y_train = train_processed["is_attributed"]

X_test = test_processed.drop("is_attributed", axis=1)
y_test = test_processed["is_attributed"]
