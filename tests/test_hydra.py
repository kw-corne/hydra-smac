import pytest
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Scenario

from src.hydrasmac.hydra import Hydra


class MockHydra:
    pass


@pytest.fixture
def config_space():
    DummyConfig = ConfigurationSpace()
    DummyConfig.add_hyperparameters(
        [
            Float("x", (1.0, 5.0)),
            Float("y", (1.0, 5.0)),
            Float("z", (1.0, 5.0)),
        ]
    )
    return DummyConfig


@pytest.fixture
def instances():
    return ["a", "b", "c"]


@pytest.fixture
def instance_features(instances):
    features = {inst: [i] for i, inst in enumerate(instances)}
    return features


@pytest.fixture
def target_function():
    def tf(config: Configuration, instance: str, seed: int = 0):
        return 1

    return tf


@pytest.fixture
def scenario(config_space, instances, instance_features):
    return Scenario(
        configspace=config_space,
        instances=instances,
        instance_features=instance_features,
        deterministic=True,
    )


@pytest.fixture
def hydra(scenario, target_function):
    return Hydra(
        scenario,
        target_function,
        hydra_iterations=4,
        smac_runs_per_iter=2,
        incumbents_added_per_iter=2,
        stop_early=True,
    )


# def test_idk(hydra: Hydra):
#     assert hydra._hydra_iterations == 5
