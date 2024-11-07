from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import get_tweeter_raw_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            
            node(
                get_tweeter_raw_data,
                ["params:url_tweeter_data"],
                "tweeter_raw_data",
                name="download_raw_data",
            ),
        ],
        tags="raw_data",
    )