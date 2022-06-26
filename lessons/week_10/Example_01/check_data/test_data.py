import pandas as pd
import scipy.stats

# Non Deterministic Test
def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    numerical_columns = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week"
    ]
    
    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

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
        
# Determinstic Test
def test_column_presence_and_type(data):
    
    # Disregard the reference dataset
    _, df = data

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
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"

# Deterministic Test
def test_class_names(data):
    
    # Disregard the reference dataset
    _, df = data

    # Check that only the known classes are present
    known_classes = [
        " <=50K",
        " >50K"
    ]

    assert df["high_income"].isin(known_classes).all()

# Deterministic Test
def test_column_ranges(data):
    
    # Disregard the reference dataset
    _, df = data

    ranges = {
        "age": (17, 90),
        "fnlwgt": (1.228500e+04, 1.484705e+06),
        "education_num": (1, 16),
        "capital_gain": (0, 99999),
        "capital_loss": (0, 4356),
        "hours_per_week": (1, 99)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert df[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={df[col_name].min()} and max={df[col_name].max()}"
        )