from copy import deepcopy

import pytest
from pytest import MonkeyPatch
from smac import Scenario

from hydrasmac.hydra.hydra import Hydra
from hydrasmac.hydra.incumbents import Incumbent, Incumbents
from hydrasmac.hydra.types import TargetFunction


@pytest.fixture
def mock_hydra(
    hydra: Hydra,
    incumbent: Incumbent,
    incumbent2: Incumbent,
    monkeypatch: MonkeyPatch,
) -> Hydra:
    def mocked_smac_runs() -> Incumbents:
        incs = [
            deepcopy(incumbent),
            deepcopy(incumbent2),
        ]

        return Incumbents(incs)

    monkeypatch.setattr(hydra, "_do_smac_runs", mocked_smac_runs)

    return hydra


def test_portfolio_len(mock_hydra: Hydra):
    portfolio = mock_hydra.optimize()

    assert len(portfolio) == mock_hydra._incumbents_added_per_iter


def test_portfolio_len_no_stop_early(mock_hydra: Hydra):
    mock_hydra._stop_early = False
    portfolio = mock_hydra.optimize()

    assert len(portfolio) == mock_hydra._incumbents_added_per_iter


def test_incs_added(target_function: TargetFunction, scenario: Scenario):
    with pytest.raises(ValueError):
        hydra = Hydra(
            target_function,
            scenario,
            incumbents_added_per_iter=10,
            smac_runs_per_iter=1,
        )
