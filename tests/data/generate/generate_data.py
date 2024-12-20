import numpy as np
from supsmu import supsmu


def generate_sin(save=False):
    """
    Generate test data with single frequency, evenly spaced, uniform noise sine wave.
    """
    np.random.seed(9345)
    n = 1000
    cycles = 2
    noise_level = 0.3
    x = np.linspace(0, 1, n, dtype=np.float32)
    y = np.sin(cycles * 2 * np.pi * x)

    bound = np.sqrt(3 * noise_level**2)
    y_noisy = y + (np.random.uniform(-bound, bound, n).astype(np.float32))

    y_smooth = supsmu(x, y_noisy)

    if save:
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            "tests/data/test_sin.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth


def generate_mixed_freq_sin(save=False):
    """
    Generate test data with mixed frequency, unevenly spaced sine wave.
    """
    np.random.seed(9345)
    n = 1000
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.zeros_like(x)

    # Low freq
    mask1 = x < 0.3
    cycles1 = 2
    y[mask1] = np.sin(cycles1 * 2 * np.pi * x[mask1])

    # High freq
    mask2 = (x >= 0.3) & (x < 0.6)
    cycles2 = 5
    y[mask2] = np.sin(cycles2 * 2 * np.pi * x[mask2])

    # Mix freq
    mask3 = x >= 0.6
    y[mask3] = np.sin(cycles1 * 2 * np.pi * x[mask3]) + 0.5 * np.sin(
        cycles2 * 2 * np.pi * x[mask3]
    )

    noise_level = 0.10
    noise = np.random.normal(0, noise_level, y.shape)

    y_noisy = y + noise
    y_noisy.astype(np.float32)

    y_smooth = supsmu(x.astype(np.float32), y_noisy.astype(np.float32))

    if save:
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            "tests/data/test_mixed_freq_sin.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth


def generate_mixed_noise_sin(save=False):
    """
    Generate test data with mixed noise, unevenly spaced sine wave.
    """
    np.random.seed(9345)
    n = 1000
    cycles = 2
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.sin(cycles * 2 * np.pi * x)

    noise = np.zeros_like(y)
    # Low noise
    mask1 = x < 0.3
    noise[mask1] = np.random.normal(0, 0.05, sum(mask1))

    # High noise
    mask2 = (x >= 0.3) & (x < 0.6)
    noise[mask2] = np.random.normal(0, 0.15, sum(mask2))

    # Medium noise
    mask3 = x >= 0.6
    noise[mask3] = np.random.normal(0, 0.1, sum(mask3))

    y_noisy = y + noise
    y_noisy.astype(np.float32)

    y_smooth = supsmu(x.astype(np.float32), y_noisy.astype(np.float32))

    if save:
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            "tests/data/test_mixed_noise_sin.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth


def generate_complex_sin(save=False):
    """
    Generate test data with mixed frequency, unevenly spaced sine wave with mixed noise.
    """
    np.random.seed(9345)
    n = 1000
    x = np.sort(np.random.uniform(0, 1, n))
    y = np.zeros_like(x)

    # Low freq
    mask1 = x < 0.3
    cycles1 = 2
    y[mask1] = np.sin(cycles1 * 2 * np.pi * x[mask1])

    # High freq
    mask2 = (x >= 0.3) & (x < 0.6)
    cycles2 = 5
    y[mask2] = np.sin(cycles2 * 2 * np.pi * x[mask2])

    # Mix freq
    mask3 = x >= 0.6
    y[mask3] = np.sin(cycles1 * 2 * np.pi * x[mask3]) + 0.5 * np.sin(
        cycles2 * 2 * np.pi * x[mask3]
    )

    noise = np.zeros_like(y)
    # Low noise
    noise[mask1] = np.random.normal(0, 0.05, sum(mask1))
    # High noise
    noise[mask2] = np.random.normal(0, 0.15, sum(mask2))
    # Medium noise
    noise[mask3] = np.random.normal(0, 0.1, sum(mask3))

    y_noisy = y + noise
    y_noisy.astype(np.float32)

    y_smooth = supsmu(x.astype(np.float32), y_noisy.astype(np.float32))

    if save:
        arr = np.asarray([x, y, y_noisy, y_smooth]).T
        np.savetxt(
            "tests/data/test_complex_sin.csv",
            X=arr,
            delimiter=",",
            newline="\n",
            comments="",
            header="x,y,y_noisy,y_smooth",
        )
    return x, y, y_noisy, y_smooth



def generate_periodic(save=False):
    pass


def generate_x_repeated(save=False):
    pass


def main():
    generate_sin(save=True)
    generate_mixed_freq_sin(save=True)
    generate_mixed_noise_sin(save=True)
    generate_complex_sin(save=True)


if __name__ == "__main__":
    main()
