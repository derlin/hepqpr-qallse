
from .data_structures import *
from .qallse_mp import QallseMp, MpConfig
from .utils import define_circle


class D0Config(MpConfig):
    d0_denom = 3.0
    d0_factor = 0.5
    z0_denom = 1.0
    z0_factor = 0.1

    # longitudinal width of the luminous region in trackml
    beamspot_width = 55 / 2.0
    # transverse width (σx,σy) = (15μm, 15μm)
    # beamspot_height = 15
    # coordinate of the luminous region center
    beamspot_center = (0, 0, 0)


class QallseD0(QallseMp):
    """ Same as QallseMp, but use a variable bias weight derived from the impact parameter."""
    config: D0Config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_base_config(self):
        return D0Config()

    def _compute_weight(self, tplet: Triplet) -> float:
        # Just return a constant for now.
        # In the future, it would be interesting to try to measure a-priori how interesting a triplet is
        # (using for example the number of quadruplets it is part of) and encode this information into a
        # variable weight.
        tplet.d0, tplet.z0 = self._compute_impact_params_for(tplet)
        # d0
        w = self.config.d0_factor * (1.0 - np.exp(-abs(tplet.d0) / self.config.d0_denom))
        # z0
        w += self.config.z0_factor * (1.0 - np.exp(-tplet.z0 / self.config.z0_denom))
        return w

    def _compute_impact_params_for(self, tplet: Triplet) -> (float, float):
        #: circle
        # tplet.circle = define_circle(*[h.coord_2d for h in tplet.hits])
        tplet.circle = define_circle(tplet.d1.h1.coord_2d, tplet.d1.h2.coord_2d, tplet.d2.h2.coord_2d)
        (cx, cy), cr = tplet.circle
        # d0, max distance between the circle and the beamspot in the transverse plane,
        # here considered to be (0,0)
        ox, oy, _ = self.config.beamspot_center
        d0 = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2) - cr

        # projection of each doublet on the Z axis
        z0_1 = abs(tplet.d2.h1.z - tplet.d1.dz / tplet.d1.dr * tplet.d2.h1.r)
        z0_2 = abs(tplet.d2.h2.z - tplet.d1.dz / tplet.d1.dr * tplet.d2.h2.r)

        # we want both projections to be inside the luminous region.
        # if so, dz0 is 0. If not, it is set to the max distance of the projection
        maxZ = np.max([z0_1, z0_2, self.config.beamspot_width]) - self.config.beamspot_width
        # actually, don't just take the max, but also look at the rz_angle of the doublets
        # TODO: why d1 and why sin ?
        z0 = abs(maxZ * math.sin(tplet.d1.rz_angle))

        return d0, z0
