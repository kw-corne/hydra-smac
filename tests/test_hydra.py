from copy import deepcopy
from pathlib import Path

import pytest
from ConfigSpace import Configuration
from pytest import MonkeyPatch
from smac import Scenario

from hydrasmac.hydra.hydra import Hydra
from hydrasmac.hydra.incumbents import Incumbent, Incumbents
from hydrasmac.hydra.types import TargetFunction


@pytest.fixture
def MockHydra(
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


def test_portfolio_len(MockHydra: Hydra):
    portfolio = MockHydra.optimize()

    assert len(portfolio) == MockHydra._incumbents_added_per_iter


def test_portfolio_len_no_stop_early(MockHydra: Hydra):
    MockHydra._stop_early = False
    portfolio = MockHydra.optimize()

    assert len(portfolio) == MockHydra._incumbents_added_per_iter


def test_incs_added(target_function: TargetFunction, scenario: Scenario):
    with pytest.raises(ValueError):
        Hydra(
            target_function,
            scenario,
            incumbents_added_per_iter=10,
            smac_runs_per_iter=1,
        )


def test_hydra_target_function(MockHydra: Hydra):
    portfolio = MockHydra.optimize()
    config = portfolio[0]

    assert MockHydra._hydra_target_function(config, "a") == 1.0


# TODO
def test_stop_early(MockHydra: Hydra):
    pass


def test_simple_hydra_run(
    trivial_scenario: Scenario, target_function: TargetFunction, tmp_path: Path
):
    hydra = Hydra(
        trivial_scenario,
        target_function,
        hydra_iterations=2,
        smac_runs_per_iter=2,
        incumbents_added_per_iter=1,
        stop_early=True,
        output_dir_path=tmp_path / "hydra_tests_out",
    )

    portfolio = hydra.optimize()

    assert len(portfolio) == 1

    # the target function always returns 1 so SMAC shouldn't find any
    # improvements to the default configuration
    assert (
        portfolio[0] == trivial_scenario.configspace.get_default_configuration()
    )


def test_validate(MockHydra: Hydra, portfolio: list[Configuration]):
    costs = MockHydra.validate(
        portfolio,
        instances=["a", "b"],
        instance_features={"a": [0.0], "b": [1.0]},
    )

    assert costs["a"] == 1 and costs["b"] == 1
