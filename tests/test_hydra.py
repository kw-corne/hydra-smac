import pytest
from ConfigSpace import Configuration, ConfigurationSpace, Float
from pytest import MonkeyPatch
from smac import RunHistory, Scenario

from src.hydrasmac.hydra import Hydra
from src.hydrasmac.incumbents import Incumbent, Incumbents
from src.hydrasmac.types import TargetFunction


def test_optimize(mock_hydra: Hydra):
    assert 1 == 1
