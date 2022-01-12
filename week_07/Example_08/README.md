# Instructions
In this exercise you will apply non-deterministic tests to the cleaned dataset from *Example 03*.

For most of the non-deterministic tests you need a reference dataset, and a dataset to be tested.
This is useful when retraining, to make sure that the new training dataset has a similar
distribution to the original dataset and therefore the method that was used originally is expected
to work well.

Since we do not have a new training dataset, we will compare the test dataset against the train
dataset. This is a useful trick when obtaining a new training dataset right away is not possible.

We will use the [Kolmogorov-Smirnov test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) for goodness of fit. Remember that the 2 sample KS test is used to test whether two vectors come from the same
distribution (null hypothesis), or from two different distributions (alternative hypothesis),
and it is non-parametric.

## Execution Steps

To run the project, execute the script:

```bash
mlflow run .
```

