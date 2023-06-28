import pandas as pd

from typing import Dict, Any

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def preprocessing_target(target:str) -> int:
    if target=='positive':
        return 2
    if target=='neutral':
        return 1
    if target=='negative':
        return 0

def split_dataset(dataset: pd.DataFrame, test_ratio: float) -> Dict[str, Any]:
    """
    Splits dataset into a training set and a test set.
    """
    dataset = dataset[~dataset["text"].isna()]
    X = dataset["text"]
    y = dataset["airline_sentiment"]
    y = y.apply(preprocessing_target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=40)

    return dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
