from copy import deepcopy

import pytest
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import RunHistory, Scenario

from hydrasmac import Hydra
from hydrasmac.hydra.incumbents import Incumbent, Incumbents
from hydrasmac.hydra.types import CostDict, TargetFunction


@pytest.fixture
def config_space() -> Configuration:
    config_spc = ConfigurationSpace()
    config_spc.add_hyperparameters(
        [
            Float("x", (1.0, 5.0)),
            Float("y", (1.0, 5.0)),
            Float("z", (1.0, 5.0)),
        ]
    )
    return config_spc


@pytest.fixture
def config_space2() -> Configuration:
    config_spc = ConfigurationSpace()
    config_spc.add_hyperparameters(
        [
            Float("x2", (10.0, 50.0)),
            Float("y2", (10.0, 50.0)),
            Float("z2", (10.0, 50.0)),
        ]
    )
    return config_spc


@pytest.fixture(autouse=True)
def incumbent(config_space, cost_dict) -> Incumbent:
    return Incumbent(config_space, RunHistory(), cost_dict)


@pytest.fixture(autouse=True)
def incumbent2(config_space2, cost_dict) -> Incumbent:
    return Incumbent(config_space2, RunHistory(), cost_dict)


@pytest.fixture(autouse=True)
def incumbents(
    incumbent: Incumbent, incumbent2: Incumbent, cost_dict
) -> Incumbents:
    cost_dict2 = deepcopy(cost_dict)

    cost_dict2["a"] = 400.0
    incumbent2.cost_dict = cost_dict2

    return Incumbents([incumbent, incumbent2])


@pytest.fixture
def cost_dict() -> CostDict:
    return {
        "a": 100.0,
        "b": 400.0,
        "c": 700.0,
    }


@pytest.fixture
def instances() -> list[str]:
    return ["a", "b", "c"]


@pytest.fixture
def instance_features(instances: list[str]) -> dict[str, list[float]]:
    features = {inst: [float(i)] for i, inst in enumerate(instances)}
    return features


@pytest.fixture
def target_function() -> TargetFunction:
    def tf(config: Configuration, instance: str, seed: int = 0):
        return 1

    return tf


@pytest.fixture
def scenario(config_space, instances, instance_features) -> Scenario:
    return Scenario(
        configspace=config_space,
        instances=instances,
        instance_features=instance_features,
        deterministic=True,
    )


@pytest.fixture
def hydra(scenario, target_function) -> Hydra:
    return Hydra(
        scenario,
        target_function,
        hydra_iterations=4,
        smac_runs_per_iter=2,
        incumbents_added_per_iter=2,
        stop_early=True,
    )
