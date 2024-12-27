import numpy as np
import pytest
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.packages import importr

from supsmu import supsmu

stats = importr("stats")
r_supsmu = stats.supsmu


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(9345)
    yield


def compare_supsmus(x, y, periodic, **kwargs):
    py_result = supsmu(x, y, periodic=periodic, **kwargs)

    np_converter = default_converter + numpy2ri.converter
    with np_converter.context():
        r_result = np.array(r_supsmu(x, y, periodic=periodic, **kwargs)["y"])

    assert np.allclose(py_result, r_result, atol=1e-2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wt": np.random.uniform(0, 1, size=1000)},
        {"span": 0.15},
        {"bass": 8},
    ],
)
def test_supsmu_core(test_data, kwargs):
    x, y, periodic = test_data
    compare_supsmus(x, y, periodic, **kwargs)


def test_supsmu_edge(test_edge_data):
    x, y, periodic = test_edge_data
    compare_supsmus(x, y, periodic)
