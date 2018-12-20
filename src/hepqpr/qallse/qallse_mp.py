import time

import pandas as pd

from .data_structures import *
from .qallse import Qallse, Config, Config1GeV
from .qallse_base import QallseBase


class MpConfig(Config1GeV):
    min_qplet_path = 2


class QallseMp(Qallse):
    config: MpConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_config(self):
        return MpConfig()

    def _find_max_path(self, qplet: Quadruplet, direction=0):
        # TODO: store those information in the quadruplet structure
        inner_length = 0
        outer_length = 0

        if direction <= 0 and len(qplet.t1.inner):
            ls = (self._find_max_path(q, direction=-1) for q in qplet.t1.inner)
            inner_length = max(ls) if len(qplet.t1.inner) > 1 else next(ls)
        if direction >= 0 and len(qplet.t2.outer):
            ls = (self._find_max_path(q, direction=1) for q in qplet.t2.outer)
            outer_length = max(ls) if len(qplet.t2.outer) > 1 else next(ls)

        return 1 + inner_length + outer_length

    def build_model(self, *args, **kwargs):
        # create the model as usual
        QallseBase.build_model(self, *args, **kwargs)

        # filter quadruplets
        start_time = time.process_time()
        dropped = self._filter_quadruplets()
        exec_time = time.process_time() - start_time

        self.logger.info(
            f'MaxPath done in {exec_time:.2f}s. '
            f'doublets: {len(self.qubo_doublets)}, triplets: {len(self.qubo_triplets)}, ' +
            f'quadruplets: {len(self.quadruplets)} (dropped {dropped})')
        self.log_build_stats()

    def _filter_quadruplets(self):

        filtered_qplets = []

        for qplet in self.quadruplets:
            q_path = self._find_max_path(qplet)
            if q_path >= self.config.min_qplet_path:
                filtered_qplets.append(qplet)
                self._register_qubo_quadruplet(qplet)
            elif self.dataw.is_real_xplet(qplet.hit_ids()) == XpletType.VALID:
                self.hard_cuts_stats.append(f'qplet,{qplet},max_path,{q_path},')

        dropped = len(self.quadruplets) - len(filtered_qplets)
        self.quadruplets = filtered_qplets

        return dropped

    def _create_quadruplets(self, register_qubo=False):
        # don't register quadruplet now, do it in a second pass
        super()._create_quadruplets(False)
