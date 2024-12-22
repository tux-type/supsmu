import numpy.typing as npt
import numpy as np

def supsmu(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    wt: npt.NDArray[np.float32] | None = None,
    span: float = 0,
    periodic: bool = False,
    bass: float = 0,
) -> npt.NDArray[np.float32]: ...
