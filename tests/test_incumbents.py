from copy import deepcopy

from ConfigSpace import Float

from src.hydrasmac.incumbents import Incumbent, Incumbents


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
    assert incumbents.get_best_n(1).incumbents == [incumbent]


def test_get_configs(incumbents: Incumbents, config_space):
    assert config_space in incumbents.get_configs()
