from typing import Any

import numpy as np
from numpy.typing import NDArray as NDArray

def interpolate_data(
    source_coordinates: NDArray[np.number[Any]],
    source_values: NDArray[np.number[Any]],
    target_coordinates: NDArray[np.number[Any]],
    *,
    is_discrete: bool,
) -> NDArray[np.number[Any]]: ...
