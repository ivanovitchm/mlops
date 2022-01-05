import pytest
import wandb
import pandas as pd

# This is global so all tests are collected under the same
# run
run = wandb.init(project="week_07_data_checks", job_type="data_checks")


@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("week_07_preprocessing/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)

    return df


def test_column_presence_and_type(data):

    required_columns = {
        "age": pd.api.types.is_int64_dtype,
        "workclass": pd.api.types.is_object_dtype,
        "fnlwgt": pd.api.types.is_int64_dtype,
        "education": pd.api.types.is_object_dtype,
        "education_num": pd.api.types.is_int64_dtype,
        "marital_status": pd.api.types.is_object_dtype,
        "occupation": pd.api.types.is_object_dtype,
        "relationship": pd.api.types.is_object_dtype,
        "race": pd.api.types.is_object_dtype,
        "sex": pd.api.types.is_object_dtype,
        "capital_gain": pd.api.types.is_int64_dtype,
        "capital_loss": pd.api.types.is_int64_dtype,  
        "hours_per_week": pd.api.types.is_int64_dtype,
        "native_country": pd.api.types.is_object_dtype,
        "high_income": pd.api.types.is_object_dtype
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Check that only the known classes are present
    known_classes = [
        " <=50K",
        " >50K"
    ]

    assert data["high_income"].isin(known_classes).all()


def test_column_ranges(data):

    ranges = {
        "age": (17, 90),
        "fnlwgt": (1.228500e+04, 1.484705e+06),
        "education_num": (1, 16),
        "capital_gain": (0, 99999),
        "capital_loss": (0, 4356),
        "hours_per_week": (1, 99)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={data[col_name].min()} and max={data[col_name].max()}"
        )
