import pandas as pd
from .data_structures import *
from .qallse import Qallse, Config1GeV
from .qallse_dj import QallseDj


class VwConfig(Config1GeV):
    bias_denom = -100
    bias_power = 2


class QallseVw(Qallse):
    config: VwConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_config(self):
        return VwConfig()

    def _find_max_path(self, tplet: Triplet, direction=0):
        inner_length = 0
        outer_length = 0

        if direction <= 0 and len(tplet.inner):
            ls = (self._find_max_path(q.t1, direction=-1) for q in tplet.inner)
            inner_length = max(ls) if len(tplet.inner) > 1 else next(ls)
        if direction >= 0 and len(tplet.outer):
            ls = (self._find_max_path(q.t2, direction=1) for q in tplet.outer)
            outer_length = max(ls) if len(tplet.outer) > 1 else next(ls)

        return 1 + inner_length + outer_length

    def _compute_weight(self, tplet: Triplet):
        max_path = self._find_max_path(tplet)
        self.max_paths.append(max_path)
        return max_path ** self.config.bias_power / self.config.bias_denom

    def to_qubo(self, return_stats=False):
        self.max_paths = []
        Q = super().to_qubo(return_stats)
        self.logger.info(pd.Series(self.max_paths).describe())
        return Q