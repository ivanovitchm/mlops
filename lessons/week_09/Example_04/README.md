# Instructions
In this exercise we will write an inference pipeline.

## Execution Steps


To run the project, execute these scripts:

### Experiment 1

```bash
mlflow run . -P hydra_options="decision_tree_pipeline.decision_tree.max_depth=5"
```

### Experiment 2

```bash
mlflow run . -P hydra_options="-m decision_tree_pipeline.decision_tree.max_depth=6,8,10"
```

### Experiment 3

```bash
mlflow run . -P hydra_options="-m decision_tree_pipeline.decision_tree.max_depth=range(11,15,2)"
```

### Experiment 4

```bash
mlflow run . -P hydra_options="-m decision_tree_pipeline.decision_tree.criterion=entropy,gini decision_tree_pipeline.decision_tree.max_depth=range(5,9,2) hydra/launcher=joblib"
```

### Experiment 5

```bash
mlflow run . -P hydra_options="-m decision_tree_pipeline.decision_tree.criterion=entropy,gini decision_tree_pipeline.numerical_pipe.model=0,1,2 hydra/launcher=joblib"
```

Hydra supports also more sophisticated algorithms than grid search. Refer to the sweepers [documentation](https://hydra.cc/docs/plugins/ax_sweeper) to find out more.

### Experiment 6

```bash
mlflow run . -P hydra_options="decision_tree_pipeline.export_artifact=model_export"
```

Exporting means packaging our inference pipeline into a format that can be saved to disk and reused by downstream tasks, for example our production environment.

We can export our inference pipeline/model using `mlflow`. MLflow provides a standard format for model exports that is accepted by many downstream tools. Each export can contain multiple flavors for the same model. A **flavor** is a particular subformat for the model. A downstream tools could support some flavors but not others. Of course, the exported artifact can also be re-read by mlflow. Finally, the export contains also all the information to recreate the environment for the model with all the right versions of all the dependencies.

MLflow provides [several flavors](https://www.mlflow.org/docs/latest/models.html#built-in-model-flavors) out of the box, and can natively export models from sklearn, pytorch, Keras, ONNX and also a generic python function flavor that can be used for custom things.

When generating the model export we can provide two optional but important elements:

- A signature, which contains the input and output schema for the data. This allows downstream tools to catch obvious schema problems.
- Some input examples: these are invaluable for testing that everything works in downstream task

Normally MLflow figures out automatically the environment that the model need to work appropriately. However, this environment can also be [explicitly controlled](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.save_model). Finally, the exported model can be [converted](https://www.mlflow.org/docs/latest/models.html#deploy-mlflow-models) to a Docker image that provides a REST API for the model.