from .binary      import AMPIBinaryIndex
from .tomography  import (AMPITomographicIndex, AMPITwoStageIndex,
                          geometry_guided_directions, estimate_nn_distance)
from .hashing     import AMPIHashIndex
from .subspace    import AMPISubspaceIndex
from .fan         import AMPIPrincipalFanIndex

AMPIIndex = AMPITomographicIndex  # default alias

__version__ = "0.5.0"
__all__ = [
    "AMPIBinaryIndex",
    "AMPITomographicIndex",
    "AMPITwoStageIndex",
    "geometry_guided_directions",
    "estimate_nn_distance",
    "AMPIHashIndex",
    "AMPISubspaceIndex",
    "AMPIPrincipalFanIndex",
    "AMPIIndex",
]
