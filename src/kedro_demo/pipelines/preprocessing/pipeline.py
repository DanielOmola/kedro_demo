from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import clean_tweeter_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            
            node(
                clean_tweeter_data,
                ["tweeter_raw_data"],
                "tweeter_clean_data",
                name="cleaning_data",
            ),
        ],
        tags="clean_data",
    )