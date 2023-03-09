from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, cast

import numpy as np
from ConfigSpace import Configuration
from smac.facade.algorithm_configuration_facade import (
    AlgorithmConfigurationFacade,
)
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario

from src.hydrasmac.types import CostDict, Incumbent, TargetFunction

logger = logging.getLogger(__name__)


class Hydra:
    """Class to use `Hydra`_ for constructing a portfolio of configurations.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    target_function : Callable
        Cost function to be minimized
    hydra_iterations : int
        The number of Hydra iterations
    smac_runs_per_iter : int
        The number of SMAC runs that are performed at each Hydra iteration.
    incumbents_added_per_iter : int
        The number of incumbents that are added to the portfolio after each
        Hydra iteration. This value cannot be higher than the number of SMAC
        runs that are started each iteration.
    stop_early : bool
        Determines if the Hydra procedure is stopped before the maximum number
        of iterations is reached, by checking if portfolio performance has not
        improved compared to the previous iteration or if a configuration
        that is already present in the portfolio is returned by a SMAC run.

    .. _Hydra:
        https://www.cs.ubc.ca/labs/algorithms/Projects/Hydra/
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: TargetFunction,
        *,
        hydra_iterations: int = 3,
        smac_runs_per_iter: int = 2,
        incumbents_added_per_iter: int = 1,
        stop_early: bool = True,
    ):
        self._scenario = scenario
        self._target_function = target_function
        self._hydra_iterations = hydra_iterations

        if incumbents_added_per_iter > smac_runs_per_iter:
            raise ValueError(
                "The number of incumbents added per iteration cannot be larger "
                "than the number of SMAC runs per iteration"
            )

        self._smac_runs_per_iter = smac_runs_per_iter
        self._incumbents_added_per_iter = incumbents_added_per_iter
        self._stop_early = stop_early

        self._instances = self._scenario.instances
        self._instance_features = self._scenario.instance_features

        self._top_output_dir = Path(f"hydra-output-{datetime.now()}")
        self._smac_run_output_dir = self._top_output_dir / "smac_runs"
        self._valdation_run_output_dir = (
            self._top_output_dir / "validation_runs"
        )

        # Hydra iter, SMAC iter
        self._smac_run_name = "iter_{}_{}"

        # instance, config portfolio index
        self._validation_run_name = "valid_{}_{}"

    def _get_scenario_dict(self) -> dict[str, Any]:
        """Returns a dictionary version of the scenario so it can be edited
        (the scenario dataclass is frozen)"""
        config_space = copy.deepcopy(self._scenario.configspace)

        # Need to edit the output directory, but scenario dataclass is frozen
        scenario_dict = self._scenario.make_serializable(self._scenario)

        # Configspace is removed during serialization so add it back
        scenario_dict["configspace"] = config_space

        # _meta can't be in the init kwargs
        scenario_dict.pop("_meta", None)

        return scenario_dict  # type: ignore

    def _get_scenario(self, run_name: str) -> Scenario:
        """Sets the instances of the scenario to the training instances, and the
        output directory for the run."""
        scenario_dict = self._get_scenario_dict()

        scenario_dict["instances"] = self._instances
        scenario_dict["instance_features"] = self._instance_features

        scenario_dict["output_directory"] = self._smac_run_output_dir
        scenario_dict["name"] = run_name

        return Scenario(**scenario_dict)

    def _get_instance_validation_scenario(
        self, instance: str, instance_feature: list[float], run_name: str
    ) -> Scenario:
        """Sets the instances of the scenario to the specified validation
        instance, and the output directory for the run."""
        scenario_dict = self._get_scenario_dict()

        scenario_dict["instances"] = [instance]
        scenario_dict["instance_features"] = {instance: instance_feature}
        scenario_dict["max_budget"] = 1

        scenario_dict["output_directory"] = self._valdation_run_output_dir
        scenario_dict["name"] = run_name

        return Scenario(**scenario_dict)

    def _hydra_target_function(
        self, config: Configuration, instance: str, seed: int = 0
    ) -> float:
        """Defines a new metric to evaluate target function runs, by returning
        the performance of the portfolio in cases where the portfolio
        outperforms the current configuration."""
        config_cost = self._target_function(config, instance, seed)
        portfolio_cost = self._cost_per_instance[instance]

        logger.debug(
            f"Instance {instance:<40}"
            f"Config cost {config_cost:<40}"
            f"Portfolio cost: {portfolio_cost:<40}"
        )

        return float(min(config_cost, portfolio_cost))

    def _should_stop_early(self) -> bool:
        """Check if the portfolio performance has stagnated"""
        return bool(
            self._stop_early
            and self._portfolio_cost
            >= self._cost_each_iter[self._hydra_iter - 1]
        )

    # TODO: (UNSURE) Now looping over instances without considering different
    #       seeds and budgets, could lead to unfair comparisons?
    def _average_cost_per_instance(
        self, config: Configuration, runhistory: RunHistory
    ) -> CostDict:
        instance_costs: dict[str, list[float]] = {
            k: [] for k in self._instances  # type: ignore
        }

        for isb in runhistory.get_instance_seed_budget_keys(config):
            if isb.instance is None:
                continue

            avg_cost = runhistory.average_cost(config, [isb], normalize=True)
            avg_cost = cast(float, avg_cost)  # no m.o. support atm

            instance_costs[isb.instance].append(avg_cost)

        mean_instance_costs: CostDict = {}

        for instance, costs in instance_costs.items():
            mean_instance_costs[instance] = (
                float(np.mean(costs)) if costs else np.nan
            )

            logger.debug(
                f"Instance: {instance:<30} Cost {instance_costs[instance]:<30}"
            )

        return mean_instance_costs

    def _do_smac_runs(self) -> list[Incumbent]:
        """Starts the SMAC runs and returns the incumbent configurations,
        corresponding runhistories and incumbent cost found by the runs.
        """
        incumbents: list[Incumbent] = []

        if self._hydra_iter == 0:
            target_function = self._target_function
        else:
            target_function = self._hydra_target_function

        for smac_iter in range(self._smac_runs_per_iter):
            run_name = self._smac_run_name.format(self._hydra_iter, smac_iter)

            smac = AlgorithmConfigurationFacade(
                scenario=self._get_scenario(run_name),
                target_function=target_function,
            )

            incumbent = smac.optimize()

            # Same configuration should not be in the portfolio more than once
            if incumbent not in [tup[0] for tup in incumbents]:
                runhistory = smac.runhistory
                incumbent_cost = self._average_cost_per_instance(
                    incumbent, runhistory
                )
                incumbents.append((incumbent, runhistory, incumbent_cost))

        return incumbents

    def _remove_duplicate_configs(self, incumbents: list[Incumbent]) -> bool:
        """Remove incumbent configurations already present in the portfolio,
        returns if duplicated were removed"""
        before_len = len(incumbents)

        incumbents[:] = [
            tup for tup in incumbents if tup[0] not in self.portfolio
        ]

        return len(incumbents) < before_len

    def _get_best_incumbents(
        self, incumbents: list[Incumbent]
    ) -> list[Incumbent]:
        """Sort and return top incumbents"""
        return sorted(
            incumbents, key=lambda x: np.nanmean(list(x[2].values()))  # type: ignore # noqa
        )[: self._incumbents_added_per_iter]

    def _update_portfolio_cost(self):
        """Updates the mean cost of the portfolio"""
        self._portfolio_cost = float(
            np.mean(list(self._cost_per_instance.values()))
        )
        self._cost_each_iter.append(self._portfolio_cost)

    def _update_cost_per_instance(self, incumbents: list[Incumbent]):
        """Update the cost per instance based on the new incumbents and
        evaluated instances. Unevaluated instances get the mean instead."""
        min_costs = []

        for _, _, cost_dict in incumbents:
            for instance, cost in cost_dict.items():
                min_cost = min(self._cost_per_instance[instance], cost)
                self._cost_per_instance[instance] = min_cost

                if min_cost != math.inf:
                    min_costs.append(min_cost)

        # Need to initialize unevaluated instances in the first iteration
        if self._hydra_iter == 0:
            mean_cost = np.mean(min_costs)

            for instance, cost in self._cost_per_instance.items():
                if cost == math.inf:
                    self._cost_per_instance[instance] = float(mean_cost)

    def _add_new_incumbents_to_portfolio(self, incumbents: list[Incumbent]):
        """Extends the portfolio with new configs, without adding duplicates,
        if duplicate(s) are found, they are returned. The new cost of the
        portfolio is also computed"""
        for config, _, _ in incumbents:
            self.portfolio.append(config)

        self._update_cost_per_instance(incumbents)
        self._update_portfolio_cost()

    def _update_portfolio(
        self,
        incumbents: list[Incumbent],
    ):
        """Update the portfolio with the incumbents and compute the new
        cost of the portfolio
        """
        had_duplicates = self._remove_duplicate_configs(incumbents)

        if had_duplicates:
            logger.info(
                "SMAC runs returned configurations already present in the "
                f"portfolio in iteration {self._hydra_iter}"
            )

        incumbents = self._get_best_incumbents(incumbents)
        self._add_new_incumbents_to_portfolio(incumbents)

    def optimize(self) -> list[Configuration]:
        """
        Constructs a new portfolio of configurations

        Returns
        -------
        portfolio : list[Configuration]
            The portfolio of configurations
        """
        self.portfolio: list[Configuration] = []
        self._cost_per_instance: DefaultDict[str, float] = defaultdict(
            lambda: math.inf
        )
        self._cost_each_iter: list[float] = []
        self._hydra_iter: int = 0

        for self._hydra_iter in range(self._hydra_iterations):
            logger.info(f"Starting Hydra iteration {self._hydra_iter}")

            incumbents = self._do_smac_runs()
            self._update_portfolio(incumbents)

            logger.info(
                f"Cost after iteration {self._hydra_iter} "
                f"is {self._cost_each_iter[self._hydra_iter]}"
            )

            if self._hydra_iter > 0 and self._should_stop_early():
                logger.info(
                    f"Performance stagnated after iteration {self._hydra_iter},"
                    " terminating..."
                )
                break

        logger.debug(f"{'Iteration':<10} {'Cost':<25}")
        logger.debug("=" * 35)
        for i, cost in enumerate(self._cost_each_iter):
            logger.debug(f"{i:<10} {cost:<20}")
        logger.debug("=" * 35)

        return self.portfolio

    def validate(
        self,
        portfolio: list[Configuration],
        instances: list[str],
        instance_features: dict[str, list[float]],
    ) -> float:
        """
        Validate the performance of a portfolio on the validation instances

        Parameters
        ----------
        portfolio : list[Configuration]
            The list of configurations to be validated on the test instances
        instances : list[str]
            The names of the instances to validate, e.g. name of a dataset
        instance_features : dict[str, list[float]] | None, defaults to None
            The (meta) features of each instance, e.g. average

        Returns
        -------
        cost : float
            The mean cost of the portfolio across the validation instances
        """
        cost_per_instance: DefaultDict[str, float] = defaultdict(
            lambda: math.inf
        )

        for i, config in enumerate(portfolio):
            for instance in instances:
                run_name = self._validation_run_name.format(instance, i)
                scenario = self._get_instance_validation_scenario(
                    instance, instance_features[instance], run_name
                )

                # TODO: Make this somehow only run configs once per instance?
                smac = MultiFidelityFacade(
                    scenario=scenario,
                    target_function=self._target_function,
                )

                # Prevents RuntimeError about calling __post_init__ first
                smac._optimizer._intensifier.__post_init__()

                cost = self._target_function(config, instance, 0)
                cost_per_instance[instance] = min(
                    cost_per_instance[instance], cost
                )

        return float(np.mean(list(cost_per_instance.values())))
