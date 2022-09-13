"""Functions to help format new datasets."""
from sktime.datasets import load_from_tsfile_to_dataframe


kepler = load_from_tsfile_to_dataframe("C:/Users/Tony/OneDrive - University of East "
                                       "Anglia/Research/Data/KeplerLightCurves/KeplerLightCurves.ts")

print(kepler.iloc[0])