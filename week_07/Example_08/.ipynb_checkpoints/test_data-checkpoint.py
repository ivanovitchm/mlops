import pytest
import wandb
import pandas as pd
import scipy.stats

# This is global so all tests are collected under the same
# run
run = wandb.init(project="week_07_data_checks", job_type="data_checks")

@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("week_07_data_segregation/train_data.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("week_07_data_segregation/test_data.csv:latest").file()
    sample2 = pd.read_csv(local_path)
    return sample1, sample2

def test_kolmogorov_smirnov(data):

    sample1, sample2 = data

    numerical_columns = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week"
    ]
    
    # Let's decide the Type I error probability (related to the False Positive Rate)
    alpha = 0.05
    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        # two-sided: The null hypothesis is that the two distributions are identical
        # the alternative is that they are not identical.
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime
