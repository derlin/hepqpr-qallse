import numpy as np


class SeedingConstants:
    zTolerance = 3.0
    maxEta = 2.7  # pseudo-rapidity
    maxDoubletLength = 300.0 #200.0 # LL: longer, since we added a volume !
    minDoubletLength = 10.0
    maxOuterRadius = 550.0
    doPSS = False
    nLayers = 8
    nPhiSlices = 53
    zMinus = -350
    zPlus = 350
    maxTheta = 2 * np.arctan(np.exp(-maxEta))
    # ctg can be used to get the max z given the max r of a doublet (or hit?)
    maxCtg = np.cos(maxTheta) / np.sin(maxTheta)  # cotangent of the theta
    minOuterZ = zMinus - maxOuterRadius * maxCtg - zTolerance
    maxOuterZ = zPlus + maxOuterRadius * maxCtg + zTolerance

    @classmethod
    def set_zMinus(cls, zm):
        """
        Update the zMinus boundary and values depending on it
        """
        cls.zMinus = zm
        cls.minOuterZ = cls.zMinus - cls.maxOuterRadius * cls.maxCtg - cls.zTolerance

    @classmethod
    def set_zPlus(cls, zp):
        """
        Update the zPlus boundary and values depending on it
        """
        cls.zPlus = zp
        cls.maxOuterZ = cls.zPlus + cls.maxOuterRadius * cls.maxCtg + cls.zTolerance
