# Instructions
In this exercise you will experiment with different ways of deploying the exported model for online and offline inference.


## Execution Steps
First we need to fetch the production model. We are going to save it into the model directory:

```bash
wandb artifact get remote_execution/model_export:latest --root model
```

### Offline Inference

```bash
mlflow models predict -t json -i model/input_example.json -m model
```

Using a ``csv`` file as input data. 


```bash
wandb artifact get remote_execution/test_data.csv:latest
```

```bash
mlflow models predict \
                -t csv \
                -i artifacts/test_data.csv:v0/test_data.csv \
                -m model
```

### Online Inference

We can serve a model for online inference by using ``mlflow models serve`` (we assume we already have our inference artifact in the model directory):

```bash
mlflow models serve -m model
```

Mlflow will create a REST API for us that we can interrogate by sending requests to the endpoint (which by default is http://localhost:5000/invocations). For example, we can do this from python like this:

```python
import requests
import json

with open("model/input_example.json") as fp:
    data = json.load(fp)

results = requests.post("http://localhost:5000/invocations", json=data)

print(results.json())
```