# Instructions
In this exercise you will create a MLflow component that preprocess the data. 
It implements in a reusable way the steps we implemented in the EDA notebook in the previous exercise (EDA Example 01).

## Run Steps

```bash
mlflow run . -P input_artifact="week_07_eda/raw_data.csv:latest" \
             -P artifact_name="preprocessed_data.csv" \
             -P artifact_type="clean_data" \
             -P artifact_description="Data after preprocessing"
```