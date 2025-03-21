from aeon.datasets import load_from_ts_file

from tsml_eval._wip.unequal_length._pad import Padder


def plot_unequal():
    X, y = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/UnivariateTS/GunPoint/GunPoint_TRAIN.ts",
    )
    X_resized, y_resized = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/GunPointResized/GunPointResized_TRAIN.ts",
    )
    X_trunc_start, y_trunc_start = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/GunPointTruncStart/GunPointTruncStart_TRAIN.ts",
    )
    X_trunc_end, y_trunc_end = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/GunPointTruncEnd/GunPointTruncEnd_TRAIN.ts",
    )
    X_subsequence, y_subsequence = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/GunPointSubsequence/GunPointSubsequence_TRAIN.ts",
    )

    i = 4

    import matplotlib.pyplot as plt

    # Plot the first series from all the X arrays
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(X[i][0])
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.plot(X_resized[i][0])
    plt.title('Resized')

    plt.subplot(2, 3, 3)
    plt.plot(X_trunc_start[i][0])
    plt.title('Truncate Start')

    plt.subplot(2, 3, 5)
    plt.plot(X_trunc_end[i][0])
    plt.title('Truncate End')

    plt.subplot(2, 3, 6)
    plt.plot(X_subsequence[i][0])
    plt.title('Subsequence')

    plt.tight_layout()
    plt.show()


def plot_padded():
    X, y = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/UnivariateTS/Car/Car_TRAIN.ts",
    )

    pad = Padder(add_noise=0.001)

    X_resized, y_resized = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/CarResized/CarResized_TRAIN.ts",
    )
    #X_resized = pad.fit_transform(X_resized)
    X_trunc_start, y_trunc_start = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/CarTruncStart/CarTruncStart_TRAIN.ts",
    )
    #X_trunc_start = pad.fit_transform(X_trunc_start)
    X_trunc_end, y_trunc_end = load_from_ts_file(
        f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/CarTruncEnd/CarTruncEnd_TRAIN.ts",
    )
    #X_trunc_end = pad.fit_transform(X_trunc_end)

    i = 1

    import matplotlib.pyplot as plt

    # Plot the first series from all the X arrays
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(X[i][0], color='firebrick', linewidth=2.5)
    plt.title('Original', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.subplot(2, 2, 2)
    plt.plot(X_resized[i][0], color='firebrick', linewidth=2.5)
    plt.title('Resized', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.subplot(2, 2, 3)
    plt.plot(X_trunc_start[i][0], color='firebrick', linewidth=2.5)
    plt.title('Truncate Start', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.subplot(2, 2, 4)
    plt.plot(X_trunc_end[i][0], color='firebrick', linewidth=2.5)
    plt.title('Truncate End', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add a common/shared axis label for the entire figure
    fig.text(0.5, 0.04, 'Timepoints', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.show()

if __name__ == "__main__":
    #plot_unequal()
    plot_padded()
