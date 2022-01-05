# Instructions
An important aspect when using ``pytest`` is understanding the fixture's scope works. 

The scope of the fixture can have a few legal values, described [here](https://docs.pytest.org/en/6.2.x/fixture.html#fixture-scopes). We are going to consider only **session** and **function**: with the former, the fixture is executed only once in a pytest session and the value it returns is used for all the tests that need it; with the latter, every test function gets a fresh copy of the data. This is useful if the tests modify the input in a way that make the other tests fail, for example.

Let's see this more closely in this exercise.

## Run Steps

To run the exercise, execute:

```bash
pytest . -vv
```