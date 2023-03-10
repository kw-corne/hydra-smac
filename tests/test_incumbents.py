from copy import deepcopy

import pytest
from ConfigSpace import ConfigurationSpace, Float
from smac.runhistory.runhistory import RunHistory

from src.hydrasmac.incumbents import Incumbent, Incumbents


@pytest.fixture()
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
def cost_dict():
    return {
        "a": 100.0,
        "b": 400.0,
        "c": 700.0,
    }


@pytest.fixture(autouse=True)
def incumbent(config_space, cost_dict):
    return Incumbent(config_space, RunHistory(), cost_dict)


@pytest.fixture(autouse=True)
def incumbents(incumbent: Incumbent, cost_dict):
    incumbent2 = deepcopy(incumbent)
    cost_dict2 = deepcopy(cost_dict)

    cost_dict2["a"] = 400.0
    incumbent2.cost_dict = cost_dict2

    return Incumbents([incumbent, incumbent2])


def test_mean_cost(incumbent: Incumbent):
    assert incumbent.mean_cost() == 400.0


def test_add_new_incumbent(incumbents: Incumbents, incumbent: Incumbent):
    before_len = len(incumbents)

    # Duplicate config so shouldn't get appended
    incumbents.add_new_incumbent(incumbent)
    assert len(incumbents) == before_len

    new_inc = deepcopy(incumbent)
    new_inc.config.add_hyperparameters([Float("zzz", (1.0, 5.0))])

    was_added = incumbents.add_new_incumbent(new_inc)
    assert was_added


def test_get_best_n(incumbents: Incumbents, incumbent: Incumbent):
    assert incumbents.get_best_n(1) == [incumbent]


def test_get_configs(incumbents: Incumbents, config_space):
    assert config_space in incumbents.get_configs()
