# -*- coding: utf-8 -*-
"""
List of datasets available from the http://tseregression.org/ archive, described in [1].

[1] Tan, C.W., Bergmeir, C., Petitjean, F. et al. Time series extrinsic regression.
Data Min Knowl Disc 35, 1032–1060 (2021). https://doi.org/10.1007/s10618-021-00745-9
"""

__author__ = ["Tony Bagnall"]

""" 14 Regression equal length no missing problems [1]"""
monash_regression = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia",
]

""" 19 Regression problems from [1] with interpolated missing and truncated unequal"""
monash_regression_nm_eq = [
    "AppliancesEnergy",
    "AustraliaRainfall",
    "BeijingPM10Quality-no-missing",
    "BeijingPM25Quality-no-missing",
    "BenzeneConcentration-no-missing",
    "BIDMC32HR",
    "BIDMC32RR",
    "BIDMC32SpO2",
    "Covid3Month",
    "FloodModeling1",
    "FloodModeling2",
    "FloodModeling3",
    "HouseholdPowerConsumption1-no-missing",
    "HouseholdPowerConsumption2-no-missing",
    "IEEEPPG",
    "LiveFuelMoistureContent",
    "NewsHeadlineSentiment",
    "NewsTitleSentiment",
    "PPGDalia-equal-length",
]
