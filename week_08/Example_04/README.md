# Instructions
In this exercise we will write an inference pipeline.

## Execution Steps


To run the project, execute this scripts:

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