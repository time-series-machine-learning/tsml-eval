import numpy as np
from aeon.datasets import load_from_ts_file, write_to_ts_file
from aeon.datasets.tsc_datasets import univariate_equal_length
from sklearn.utils import check_random_state


def resize_series(series, new_size, rng):
    x_new = np.zeros((series.shape[0], new_size))
    x2 = np.linspace(0, 1, series.shape[1])
    x3 = np.linspace(0, 1, new_size)
    for i, row in enumerate(series):
        x_new[i] = np.interp(x3, x2, row)
    return x_new


def truncate_series_start(series, new_size, rng):
    return series[:, -new_size:]


def truncate_series_end(series, new_size, rng):
    return series[:, :new_size]


def series_subsequence(series, new_size, rng):
    r = series.shape[1] - new_size
    front = rng.randint(0, r)
    return series[:, front:-(r-front)]


def transform_X(X, indices, new_sizes, func, rng):
    X_new = [x for x in X]
    for i in range(len(indices)):
        X_new[indices[i]] = func(X[indices[i]], new_sizes[i], rng)
        assert X_new[indices[i]].shape[1] == new_sizes[i]
    return X_new


def create_unequal_ucr():
    for dataset in univariate_equal_length:
        print(dataset)

        X_train, y_train = load_from_ts_file(
            f"C:/Users/mattm/Documents/Work/Datasets/UnivariateTS/{dataset}/{dataset}_TRAIN.ts",
        )
        X_test, y_test = load_from_ts_file(
            f"C:/Users/mattm/Documents/Work/Datasets/UnivariateTS/{dataset}/{dataset}_TEST.ts",
        )

        rng = check_random_state(0)

        train_indices = rng.choice(len(X_train), int(0.8 * len(X_train)), replace=False)
        test_indices = rng.choice(len(X_test), int(0.8 * len(X_test)), replace=False)
        train_new_size = rng.uniform(0.2, 1, len(train_indices))
        train_new_size = [int(x * X_train[train_indices[i]].shape[1]) for i, x in enumerate(train_new_size)]
        test_new_size = rng.uniform(0.2, 1, len(test_indices))
        test_new_size = [int(x * X_test[test_indices[i]].shape[1]) for i, x in enumerate(test_new_size)]

        X_train_resize = transform_X(X_train, train_indices, train_new_size, resize_series, rng)
        write_to_ts_file(
            X_train_resize,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}Resized/",
            y=y_train,
            problem_name=f"{dataset}Resized_TRAIN.ts",
        )
        X_test_resize = transform_X(X_test, test_indices, test_new_size, resize_series, rng)
        write_to_ts_file(
            X_test_resize,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}Resized/",
            y=y_test,
            problem_name=f"{dataset}Resized_TEST.ts",
        )

        X_train_truncate_start = transform_X(X_train, train_indices, train_new_size, truncate_series_start, rng)
        write_to_ts_file(
            X_train_truncate_start,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}TruncStart/",
            y=y_train,
            problem_name=f"{dataset}TruncStart_TRAIN.ts",
        )
        X_test_truncate_start = transform_X(X_test, test_indices, test_new_size, truncate_series_start, rng)
        write_to_ts_file(
            X_test_truncate_start,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}TruncStart/",
            y=y_test,
            problem_name=f"{dataset}TruncStart_TEST.ts",
        )

        X_train_truncate_end = transform_X(X_train, train_indices, train_new_size, truncate_series_end, rng)
        write_to_ts_file(
            X_train_truncate_end,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}TruncEnd/",
            y=y_train,
            problem_name=f"{dataset}TruncEnd_TRAIN.ts",
        )
        X_test_truncate_end = transform_X(X_test, test_indices, test_new_size, truncate_series_end, rng)
        write_to_ts_file(
            X_test_truncate_end,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}TruncEnd/",
            y=y_test,
            problem_name=f"{dataset}TruncEnd_TEST.ts",
        )

        X_train_subsequence = transform_X(X_train, train_indices, train_new_size, series_subsequence, rng)
        write_to_ts_file(
            X_train_subsequence,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}Subsequence/",
            y=y_train,
            problem_name=f"{dataset}Subsequence_TRAIN.ts",
        )
        X_test_subsequence = transform_X(X_test, test_indices, test_new_size, series_subsequence, rng)
        write_to_ts_file(
            X_test_subsequence,
            f"C:/Users/mattm/Documents/Work/Datasets/WorkingArea/UnequalLength/UCR/{dataset}Subsequence/",
            y=y_test,
            problem_name=f"{dataset}Subsequence_TEST.ts",
        )


if __name__ == "__main__":
    create_unequal_ucr()
