# -*- coding: utf-8 -*-
"""Functions to help format new datasets."""
from sktime.datasets import (
    load_from_tsfile_to_dataframe,
    load_UCR_UEA_dataset,
    write_tabular_transformation_to_arff,
)

tiselacX, tiselacY = load_UCR_UEA_dataset(
    "Tiselac", split="Train", extract_path="C:\\Temp\\"
)


print(tiselacX.iloc[0])
