import numpy.typing as npt
import numpy as np

def supsmu(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    wt: npt.NDArray[np.floating] | None = None,
    span: float = 0,
    periodic: bool = False,
    bass: float = 0,
) -> npt.NDArray[np.float64]: ...
