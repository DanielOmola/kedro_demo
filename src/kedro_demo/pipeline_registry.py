"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro_demo.pipelines.download import pipeline as downloading_pipeline
from kedro_demo.pipelines.preprocessing import pipeline as cleaning_tweeter_data
from kedro_demo.pipelines.split import pipeline as split_data
from kedro_demo.pipelines.training import pipeline as training

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    downloading_data = downloading_pipeline.create_pipeline()
    cleaning_data = cleaning_tweeter_data.create_pipeline()
    spliting_data = split_data.create_pipeline()
    training_model = training.create_pipeline()

    return {"__default__": downloading_data + cleaning_data + spliting_data + training_model,
            "downloading_data": downloading_data,
            "cleaning_data": cleaning_data,
            "spliting_data": spliting_data,
            "training":training_model,
            "process_train": cleaning_data + spliting_data + training_model

            }
