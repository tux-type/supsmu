import numpy as np
from supsmu import supsmu


def generate_sin(seed=9345, periodic=True, save=False):
    """
    Generate test data with single frequency, evenly spaced, uniform noise sine wave.
    """
    np.random.seed(seed)
    n = 1000
    cycles = 2
    noise_level = 0.3
    x = np.linspace(0, 1, n, dtype=np.float32)
    y = np.sin(cycles * 2 * np.pi * x)

    if not periodic:
        y = y + 2 * x

    bound = np.sqrt(3 * noise_level**2)
    y_noisy = y + (np.random.uniform(-bound, bound, n).astype(np.float32))

    y_smooth = supsmu(x, y_noisy)

    if save:
        func_type = "periodic" if periodic else "aperiodic"
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            f"tests/data/test_sin_{func_type}.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth


def generate_complex_sin(seed=9345, periodic=True, save=False):
    """
    Generate test data with mixed frequency, unevenly spaced sine wave with mixed noise.
    """
    np.random.seed(seed)
    n = 1000
    x = np.sort(np.random.uniform(0, 1, n))

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
    y_noisy.astype(np.float32)

    y_smooth = supsmu(x.astype(np.float32), y_noisy.astype(np.float32))

    if save:
        func_type = "periodic" if periodic else "aperiodic"
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            f"tests/data/test_complex_sin_{func_type}.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth


def generate_all_x_same(seed=9345, save=False):
    np.random.seed(seed)
    n = 100
    x = np.ones(n, dtype=np.float32)
    y_noisy = np.random.uniform(low=0, high=2, size=n)

    if save:
        arr = np.asarray([x, y_noisy, y_noisy]).T
        np.savetxt(
            "tests/data/test_all_x_same.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy",
        )


def generate_some_x_same(seed=9345, save=False):
    np.random.seed(seed)
    n = 100
    x = np.sort(np.round(np.random.uniform(0, 1, n)))
    y = np.sin(2 * np.pi * x)

    y_noisy = y + np.random.normal(0, 0.15, n)

    if save:
        arr = np.asarray([x, y, y_noisy]).T
        np.savetxt(
            "tests/data/test_some_x_same.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy",
        )


def main():
    generate_sin(periodic=True, save=True)
    generate_complex_sin(periodic=True, save=True)
    generate_sin(periodic=False, save=True)
    generate_complex_sin(periodic=False, save=True)
    generate_all_x_same(save=True)
    generate_some_x_same(save=True)


if __name__ == "__main__":
    main()
