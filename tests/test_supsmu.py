import pytest
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from supsmu import supsmu
from tests.data.generate_data import generate_test_arrays, generate_test_edge_arrays

numpy2ri.activate()

stats = importr("stats")
r_supsmu = stats.supsmu


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(9345)
    yield


def compare_supsmus(x, y, periodic, kwargs):
    py_result = supsmu(x, y, periodic=periodic, **kwargs)
    r_result = np.array(r_supsmu(x, y, periodic=periodic, **kwargs))
    assert np.allclose(py_result, r_result)


@pytest.mark.parameterize("x,y,periodic", generate_test_arrays(size=1000))
@pytest.mark.parameterize(
    "kwargs",
    [
        {"wt": np.random.uniform(0, 1, size=1000)},
        {"span": 5},
        {"bass": 8},
    ],
)
def test_supsmu(x, y, periodic, kwargs):
    compare_supsmus(x, y, periodic, kwargs)


@pytest.mark.parameterize("x,y,periodic", generate_test_edge_arrays())
def test_supsmu_edge(x, y, periodic, kwargs):
    compare_supsmus(x, y, periodic, kwargs)
