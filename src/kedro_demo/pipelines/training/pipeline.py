from kedro.pipeline import Pipeline, node

from .nodes import train_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["X_train", "y_train", "X_test", "y_test",
                  "params:n_splits_StratifiedKFold"],
                "model"
            )
        ]
    )