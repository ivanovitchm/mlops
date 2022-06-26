# Instructions
In this exercise you will build a component that fetches a model and test it on the test dataset.

Then, you will mark that model as **production ready**. Verify that the AUC and the confusion matrix look good, then go to the Artifact section in W&B and add the tag `prod` to the model export artifact to mark is as `production-ready`.

## Execution Steps

To run the project, execute this script:

```bash
mlflow run . -P test_data="week_08_data_segregation/test_data.csv:latest" \
             -P model_export="week_09_example_04/model_export:v0"
```
