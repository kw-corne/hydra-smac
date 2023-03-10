from pathlib import Path
import src.util.scenario_util as scenario_util

from ConfigSpace import ConfigurationSpace, Float
from smac.scenario import Scenario

DummyConfig = ConfigurationSpace()
DummyConfig.add_hyperparameters(
    [
        Float("x", (1.0, 5.0)),
        Float("y", (1.0, 5.0)),
        Float("z", (1.0, 5.0)),
    ]
)

instances = ["a", "b", "c"]
instance_features = {"a": [0.0], "b": [0.0], "c": [0.0]}

DummyScenario = Scenario(
    DummyConfig,
    instances=instances,
    instance_features=instance_features,
)


def test_get_scenario_dict():
    scenario_dict = scenario_util.get_scenario_dict(DummyScenario)

    # (TODO) Comparing the most important keys here, perhaps add a check for
    # all relevant keys later?
    assert scenario_dict["configspace"] == DummyConfig
    assert scenario_dict["instances"] == DummyScenario.instances
    assert scenario_dict["instance_features"] == DummyScenario.instance_features


def test_set_scenario_output_dir():
    output_dir = Path("./test_dir")  # TODO
    run_name = "test_run"

    scenario = scenario_util.set_scenario_output_dir(
        DummyScenario, output_dir, run_name
    )

    assert scenario.output_directory == output_dir / run_name / str(
        scenario.seed
    )
    assert scenario.name == run_name

def test_set_scenario_instances():
    dummy_inst = ["hello", "world"]
    dummy_inst_feat = {"hello" : [2.0], "world" : [10.0]}

    scenario = scenario_util.set_scenario_instances(
        DummyScenario, dummy_inst, dummy_inst_feat
    )
    
    assert scenario.instances == dummy_inst
    assert scenario.instance_features == dummy_inst_feat
