from typing import Tuple, List, Dict

#: Type of a QUBO. A QUBO has the form Σ a_i q_i + Σ b_ij q_i q_j and is represented in Python as a dictionary.
#: The entries are either `(q_i, q_i) = a_i` or `(q_i, q_j) = b_ij`.
TQubo = Dict[Tuple, float]
#: Type of a QUBO sample (i.e. a solution). It has the form `(q_i, q_i) = 0 or 1`.
TDimodSample = Dict[Tuple, int]
#: Type of a generic xplet (doublet, triplet, subtrack...), represented by an ordered list of hit ids.
#: The order is an ascending R/radius in the X-Y plane.
TXplet = List[int]