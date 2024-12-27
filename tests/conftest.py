import pytest
import numpy as np


def generate_sine(size, periodic):
    """
    Generate test data with single frequency, evenly spaced, uniform noise sine wave.
    """
    cycles = 2
    noise_level = 0.3
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.sin(cycles * 2 * np.pi * x)

    if not periodic:
        y = y + 2 * x

    bound = np.sqrt(3 * noise_level**2)
    y_noisy = y + (np.random.uniform(-bound, bound, size).astype(np.float32))

    return x, y_noisy, periodic


def generate_complex_sine(size, periodic):
    """
    Generate test data with mixed frequency, unevenly spaced sine wave with mixed noise.
    """
    x = np.sort(np.random.uniform(0, 1, size)).astype(np.float32)

    mask1 = x < 0.3
    mask2 = (x >= 0.3) & (x < 0.6)
    mask3 = x >= 0.6

    if periodic:
        y = np.sin(2 * np.pi * x)
    else:
        y = np.zeros_like(x)
        cycles1, cycles2 = 2, 5
        y[mask1] = np.sin(cycles1 * 2 * np.pi * x[mask1])  # Lower freq.
        y[mask2] = np.sin(cycles2 * 2 * np.pi * x[mask2])  # Higher freq.
        y[mask3] = np.sin(cycles1 * 2 * np.pi * x[mask3]) + np.sin(
            cycles2 * 2 * np.pi * x[mask3]
        )  # Mix freq
        y = y + 2 * x  # Add linear trend

    noise = np.zeros_like(y)
    noise[mask1] = np.random.normal(0, 0.05, sum(mask1))
    noise[mask2] = np.random.normal(0, 0.15, sum(mask2))
    noise[mask3] = np.random.normal(0, 0.1, sum(mask3))

    y_noisy = y + noise
    y_noisy = y_noisy.astype(np.float32)

    return x, y_noisy, periodic


def generate_all_x_same(size):
    x = np.ones(size, dtype=np.float32)
    y_noisy = np.random.uniform(low=0, high=2, size=size).astype(np.float32)

    return x, y_noisy, False


def generate_some_x_same(size):
    x = np.sort(np.round(np.random.uniform(0, 1, size))).astype(np.float32)
    y = np.sin(2 * np.pi * x)

    y_noisy = y + np.random.normal(0, 0.15, size)
    y_noisy = y_noisy.astype(np.float32)
    return x, y_noisy, False


def generate_test_arrays(size=1000):
    return [
        pytest.param(generate_sine(size=size, periodic=True), id="periodic-sine"),
        pytest.param(generate_complex_sine(size=size, periodic=True), id="periodic-complex"),
        pytest.param(generate_sine(size=size, periodic=False), id="aperiodic-sine"),
        pytest.param(generate_complex_sine(size=size, periodic=False), id="aperiodic-complex"),
    ]  # fmt: skip


def generate_test_edge_arrays(size=100):
    return [
        pytest.param(generate_all_x_same(size=size), id="all-x-same"),
        pytest.param(generate_some_x_same(size=size), id="some-x-same"),
    ]


@pytest.fixture(params=generate_test_arrays())
def test_data(request):
    return request.param


@pytest.fixture(params=generate_test_edge_arrays())
def test_edge_data(request):
    return request.param
