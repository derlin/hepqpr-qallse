class DisjointSets:
    def __init__(self, items):
        n = len(items)
        self.size = n
        self.n_sets = n
        self.parents = [-1] * n

        self.i2n = dict(enumerate(items))
        self.n2i = dict(reversed(t) for t in self.i2n.items())

    def _root(self, i):
        if self.parents[i] < 0:
            return i
        r = self._root(self.parents[i])
        self.parents[i] = r
        return r

    def _are_together(self, i, j):
        return self._root(i) == self._root(j)

    def root(self, n):
        return self._root(self.n2i[n])

    def are_together(self, a, b):
        return self._are_together(self.n2i[a], self.n2i[b])

    def merge(self, a, b):
        ir, jr = self.root(a), self.root(b)
        if ir == jr:
            return
        ni, nj = self.parents[ir], self.parents[jr]
        if ni > nj:
            ir, jr = jr, ir  # now ir is the "big" set
        if ni == nj:
            self.parents[ir] -= 1
        self.parents[jr] = ir
        self.n_sets -= 1

    def all_sets(self):
        res = {}
        for i in range(0, self.size):
            ri = self._root(i)
            if ri not in res:
                res[ri] = []
            res[ri].append(self.i2n[i])
        return list(res.values())


def dj_test():
    items = [chr(i) for i in range(ord('a'), ord('a') + 11)]
    #items = list(range(10))
    dj = DisjointSets(items)
    print(dj.n_sets, dj.all_sets())
    dj.merge(items[1], items[4])
    print(dj.n_sets, dj.all_sets())
    dj.merge(items[5], items[6])
    print(dj.n_sets, dj.all_sets())
    dj.merge(items[7], items[8])
    print(dj.n_sets, dj.all_sets())
    dj.merge(items[1], items[7])
    print(dj.n_sets, dj.all_sets())