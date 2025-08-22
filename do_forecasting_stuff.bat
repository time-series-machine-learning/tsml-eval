@echo off
setlocal enabledelayedexpansion

REM Activate the virtual environment
call ..\tsml-eval-venv\Scripts\activate.bat

set SCRIPT_PATH=tsml_eval\experiments\forecasting_experiments.py
set DATA_DIR=..\aeon\aeon\datasets\local_data\differenced_forecasting
set RESULTS_DIR=..\ForecastingResults\differenced_results
set FORECASTER=AutoSARIMA
set FIXED_PARAM=0

for %%D in (
    weather_dataset_T1
    weather_dataset_T2
    weather_dataset_T3
    weather_dataset_T4
    weather_dataset_T5
    solar_10_minutes_dataset_T1
    solar_10_minutes_dataset_T2
    solar_10_minutes_dataset_T3
    solar_10_minutes_dataset_T4
    solar_10_minutes_dataset_T5
    sunspot_dataset_without_missing_values_T1
    wind_farms_minutely_dataset_without_missing_values_T1
    wind_farms_minutely_dataset_without_missing_values_T3
    wind_farms_minutely_dataset_without_missing_values_T4
    wind_farms_minutely_dataset_without_missing_values_T5
    elecdemand_dataset_T1
    us_births_dataset_T1
    saugeenday_dataset_T1
    london_smart_meters_dataset_without_missing_values_T1
    london_smart_meters_dataset_without_missing_values_T2
    london_smart_meters_dataset_without_missing_values_T3
    traffic_hourly_dataset_T1
    traffic_hourly_dataset_T2
    traffic_hourly_dataset_T3
    traffic_hourly_dataset_T4
    traffic_hourly_dataset_T5
    electricity_hourly_dataset_T1
    electricity_hourly_dataset_T2
    electricity_hourly_dataset_T3
    pedestrian_counts_dataset_T1
    pedestrian_counts_dataset_T2
    pedestrian_counts_dataset_T3
    pedestrian_counts_dataset_T4
    pedestrian_counts_dataset_T5
    kdd_cup_2018_dataset_without_missing_values_T1
    australian_electricity_demand_dataset_T1
    australian_electricity_demand_dataset_T2
    australian_electricity_demand_dataset_T3
    oikolab_weather_dataset_T1
    oikolab_weather_dataset_T2
    oikolab_weather_dataset_T3
    oikolab_weather_dataset_T4
    m4_monthly_dataset_T122
    m4_monthly_dataset_T145
    m4_monthly_dataset_T180
    m4_monthly_dataset_T186
    m4_monthly_dataset_T17051
    m4_monthly_dataset_T17088
    m4_monthly_dataset_T17132
    m4_monthly_dataset_T17146
    m4_monthly_dataset_T26710
    m4_monthly_dataset_T27138
    m4_monthly_dataset_T27170
    m4_monthly_dataset_T27175
    m4_monthly_dataset_T27186
    m4_monthly_dataset_T37009
    m4_monthly_dataset_T37070
    m4_monthly_dataset_T37238
    m4_monthly_dataset_T37248
    m4_monthly_dataset_T47915
    m4_weekly_dataset_T1
    m4_weekly_dataset_T2
    m4_weekly_dataset_T19
    m4_weekly_dataset_T20
    m4_weekly_dataset_T21
    m4_weekly_dataset_T55
    m4_weekly_dataset_T56
    m4_weekly_dataset_T60
    m4_weekly_dataset_T61
    m4_weekly_dataset_T62
    m4_weekly_dataset_T224
    m4_weekly_dataset_T225
    m4_weekly_dataset_T226
    m4_weekly_dataset_T227
    m4_weekly_dataset_T248
    m4_weekly_dataset_T249
    m4_weekly_dataset_T250
    m4_daily_dataset_T1
    m4_daily_dataset_T2
    m4_daily_dataset_T6
    m4_daily_dataset_T130
    m4_daily_dataset_T131
    m4_daily_dataset_T145
    m4_daily_dataset_T1604
    m4_daily_dataset_T1605
    m4_daily_dataset_T1606
    m4_daily_dataset_T1607
    m4_daily_dataset_T1614
    m4_daily_dataset_T1615
    m4_daily_dataset_T1634
    m4_daily_dataset_T1650
    m4_daily_dataset_T2036
    m4_daily_dataset_T2037
    m4_daily_dataset_T2041
    m4_daily_dataset_T3595
    m4_daily_dataset_T3597
    m4_hourly_dataset_T170
    m4_hourly_dataset_T171
    m4_hourly_dataset_T172
) do (
    echo Running for dataset: %%D
    python -u %SCRIPT_PATH% %DATA_DIR% %RESULTS_DIR% %FORECASTER% %%D %FIXED_PARAM%
)

endlocal