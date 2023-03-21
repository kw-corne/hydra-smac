# Hydra-SMAC 0.0.8

A minimal Python re-implementation of [Hydra](https://www.cs.ubc.ca/labs/algorithms/Projects/Hydra/).

## Getting started

```bash
pip install hydrasmac
```

## Example

For more information on how to use Scenario objects, please refer to the [SMAC](https://github.com/automl/SMAC3) documentation.

```py
from ConfigSpace import Configuration, ConfigurationSpace, Float
from hydrasmac import Hydra
from smac import Scenario

instances = ["a", "b", "c"]
features = {"a": [0.0], "b": [1.0], "c": [2.0]}

cs = ConfigurationSpace()
cs.add_hyperparameters(
    [
        Float("x", (1.0, 5.0)),
        Float("y", (1.0, 5.0)),
        Float("z", (1.0, 5.0)),
    ]
)


def target_function(config: Configuration, instance: str, seed: int = 0) -> float:
    config_dict = config.get_dictionary()
    x, y, z = config_dict["x"], config_dict["y"], config_dict["z"]

    if instance == "a" and x < 2.5 and y > 2.5 and z > 2.5:
        return 0.001

    if instance == "b" and y < 2.5 and x > 2.5 and z > 2.5:
        return 0.01

    if instance == "c" and z < 2.5 and y > 2.5 and x > 2.5:
        return 0.1

    return 1


scenario = Scenario(
    configspace=cs,
    instances=instances,
    instance_features=features,
    n_trials=500,
)

hydra = Hydra(
    scenario,
    target_function,
    hydra_iterations=3,
    smac_runs_per_iter=1,
    incumbents_added_per_iter=1,
    stop_early=True,
)

portfolio = hydra.optimize()
print("====== Resulting portfolio ======")
print(portfolio)
print("=================================")
```
