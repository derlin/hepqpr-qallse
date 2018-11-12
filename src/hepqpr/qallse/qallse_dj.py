import time

from .disjoint_sets import DisjointSets

from .qallse import Qallse


class QallseDj(Qallse):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_triplets = set()

    def _find_qplet(self, t1, t2):
        for qplet in t1.inner:
            if qplet.t1 == t2:
                return qplet
        for qplet in t1.outer:
            if qplet.t2 == t2:
                return qplet
        return None

    def build_model(self, *args, **kwargs):
        super().build_model(*args, **kwargs)

        start_time = time.perf_counter()
        # compute disjoint sets on triplets
        dj = DisjointSets(self.potential_triplets)
        for qplet in self.quadruplets:
            dj.merge(qplet.t1, qplet.t2)
        self.all_sets = dj.all_sets()

        # get the list of sets of size 2
        # a size of 2 means a quadruplet is isolated
        mini_sets = [s for s in self.all_sets if len(s) <= 2]
        self.logger.debug(f'Number of sets of size 2: {len(mini_sets)}/{len(self.all_sets)}')
        # remove all quadruplets of size 2
        qplets = set(self.quadruplets)
        dropped = 0
        for s in mini_sets:
            t1, t2 = s
            if len(s) < 2:
                self.logger.error(f'Got a disjoint set of less than two triplets: {s}')
                continue
            qplet = self._find_qplet(t1, t2)
            if qplet is not None and qplet in qplets:
                qplets.remove(qplet)
                dropped += 1

        # now, register the qubo structures to use
        self.qubo_hits, self.qubo_doublets, self.qubo_triplets = {}, set(), set()
        self.quadruplets = []
        for qplet in qplets:
            self.quadruplets.append(qplet)
            super()._register_qubo_quadruplet(qplet)
        exec_time = time.perf_counter() - start_time

        self.logger.info(
            f'DJ done in {exec_time:.2f}s. '
            f'doublets: {len(self.qubo_doublets)}, triplets: {len(self.qubo_triplets)}, ' +
            f'quadruplets: {len(self.quadruplets)} (dropped {dropped})')

    def _create_quadruplets(self, register_qubo=True):
        super()._create_quadruplets(register_qubo)

    def _register_qubo_quadruplet(self, qplet):
        self.potential_triplets.update([qplet.t1, qplet.t2])
