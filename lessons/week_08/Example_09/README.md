# Instructions
In this exercise we will modify the non-deterministic test we prepared in the previous example,
by allowing it to accept the reference dataset, the new dataset as well as the threshold
for the statistical test from the command line. This is fundamental for configurability and
reusability.

## Execution Steps

To run the project, execute the script:

```bash
mlflow run . -P reference_artifact="week_08_data_segregation/train_data.csv:latest" \
             -P sample_artifact="week_08_data_segregation/test_data.csv:latest" \
             -P ks_alpha=0.05
```