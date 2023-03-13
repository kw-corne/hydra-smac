from pathlib import Path

from ConfigSpace import Configuration
from smac.scenario import Scenario

import src.util.scenario_util as scenario_util


def test_get_scenario_dict(scenario: Scenario, config_space: Configuration):
    scenario_dict = scenario_util.get_scenario_dict(scenario)

    # (TODO) Comparing the most important keys here, perhaps add a check for
    # all relevant keys later?
    assert scenario_dict["configspace"] == config_space
    assert scenario_dict["instances"] == scenario.instances
    assert scenario_dict["instance_features"] == scenario.instance_features


def test_set_scenario_output_dir(scenario: Scenario):
    output_dir = Path("./test_dir")  # TODO
    run_name = "test_run"

    scenario = scenario_util.set_scenario_output_dir(
        scenario, output_dir, run_name
    )

    assert scenario.output_directory == output_dir / run_name / str(
        scenario.seed
    )
    assert scenario.name == run_name


def test_set_scenario_instances(
    scenario: Scenario,
    instances: list[str],
    instance_features: dict[str, list[float]],
):
    scenario = scenario_util.set_scenario_instances(
        scenario, instances, instance_features
    )

    assert scenario.instances == instances
    assert scenario.instance_features == instance_features
