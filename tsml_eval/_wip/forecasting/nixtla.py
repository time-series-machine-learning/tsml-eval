import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS
from statsforecast.utils import AirPassengersDF
from aeon.datasets import load_airline

print("Examples of ETS with NIXTLA")

print("Their airline example with default ETS set up")
df = AirPassengersDF

# Nixtla requires models as a list
sf = StatsForecast(models=[ETS()],freq='M')

sf.fit(df)
x=sf.predict(h=2, level=[95])
print(x)
print(type(x))
x=sf.predict(h=1)
print(x)
print(type(x))

print("Our airline example with default ETS set up")
from aeon.datasets import load_airline
from statsforecast import StatsForecast
sf = StatsForecast(models=[ETS(season_length=4)], freq='M')


airline = load_airline()
#a = pd.DataFrame(airline)
a = airline.to_numpy()
ets = ETS(season_length=4)
ets.fit(a)
x=ets.predict(h=1)
print(x)
print(type(x))
