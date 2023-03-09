from typing import Callable

from ConfigSpace import Configuration
from smac.runhistory.runhistory import RunHistory

CostDict = dict[str, float]
Incumbent = tuple[Configuration, RunHistory, CostDict]
TargetFunction = Callable[[Configuration, str, int], float]
