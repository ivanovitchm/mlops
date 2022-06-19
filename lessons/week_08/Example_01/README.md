# Instructions
In this guided-exercise you will perform a simple Exploratory Data Analysis in Jupyter keeping track of your progress with W&B.

Even though this step is interactive, and not based on scripts, we are still going to use MLflow
to ensure a reproducible analysis, by fixing the environment in the ``conda.yml`` file.

> **_NOTE:_**  The dataset used in this guided-exercise is based on individual income in the United States. The **data** is from the **1994 census**, and contains information on an individual's **marital status**, **age**, **type of work**, and more. The **target column**, or what we want to predict, is whether individuals make less than or equal to **50k a year**, or more than **50k a year**.

You can download the data from the [University of California, Irvine's website](http://archive.ics.uci.edu/ml/datasets/Adult).

## Run steps

```bash
mlflow run .
```

## In case of errors

When you make an error writing your ``conda.yml`` file, you might end up with an environment for the pipeline or one of the components that is corrupted. Most of the time ``mlflow`` realizes that and creates a new one every time you try to fix the problem. However, sometimes this does not happen, especially if the problem was in the ``pip`` dependencies. In that case, you might want to clean up all conda environments created by ``mlflow`` and try again. In order to do so, you can get a list of the environments you are about to remove by executing:

```bash
conda info --envs | grep mlflow | cut -f1 -d" "
```

If you are ok with that list, execute this command to clean them up:

**NOTE**: this will remove ALL the environments with a name starting with mlflow. Use at your own risk

```bash
for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y;done
```

This will iterate over all the environments created by mlflow and remove them.
