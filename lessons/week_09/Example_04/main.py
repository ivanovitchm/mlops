import json

import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def process_args(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Serialize decision tree configuration
    model_config = os.path.abspath("decision_tree_pipeline_config.yml")
    
    with open(model_config, "w+") as fp:
        fp.write(OmegaConf.to_yaml(config["decision_tree_pipeline"]))

    _ = mlflow.run(
        os.path.join(root_path, "decision_tree"),
        "main",
        parameters={
            "train_data": config["data"]["train_data"],
            "model_config": model_config,
            "export_artifact": config["decision_tree_pipeline"]["export_artifact"]
        }
    )
    

if __name__ == "__main__":
    process_args()
