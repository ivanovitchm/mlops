# Instructions
In this exercise you will release your final pipeline as a versioned code artifact on GitHub.

## Execution Steps
Go to Github and make a release with version 1.0.0.

```bash
mlflow run -v 1.0.0 [URL of your Github repo] -P ...
```

```bash
mlflow run -v 1.0.0 https://github.com/ivanovitchm/high_income.git -P hydra_options="main.project_name=remote_execution"
```

