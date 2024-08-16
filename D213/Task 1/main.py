from pmdarima.arima import ADFTest
from pmdarima.arima import auto_arima
from pylab import rcParams
from scipy.signal import periodogram
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

plt.style.use("fivethirtyeight")
plt.rcParams["lines.linewidth"] = 1.5
dark_style = {
    "figure.facecolor": "#212946",
    "axes.facecolor": "#212946",
    "savefig.facecolor": "#212946",
    "axes.grid": True,
    "axes.grid.which": "both",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "grid.color": "#2A3459",
    "grid.linewidth": "1",
    "text.color": "0.9",
    "axes.labelcolor": "0.9",
    "xtick.color": "0.9",
    "ytick.color": "0.9",
    "font.size": 12,
}
plt.rcParams.update(dark_style)

rcParams["figure.figsize"] = (18, 7)


# Load the dataset without index column
df = pd.read_csv("teleco_time_series.csv", index_col=False)

start_date = pd.to_datetime("2020-01-01")
df["Day"] = start_date + pd.to_timedelta(df["Day"] - 1, unit="D")

missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}")

df.columns = ["date", "revenue"]
df.set_index("date", inplace=True)
df_diff = df.diff().dropna()

# DATA CLEANING
# C1.  Summarize the data cleaning process by doing the following: Provide a line graph visualizing the realization of the time series.

# Plot the time series df
plt.figure(figsize=(12, 6))
plt.plot(df.index, df.revenue, label="Daily Revenue")
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue (in million dollars)")
plt.legend()
plt.show()

# C3.  Evaluate the stationarity of the time series.
# Plot the data with a trend line to evaluate stationarity

plt.figure(figsize=(12, 6))
plt.plot(df.index, df.revenue, label="Daily Revenue")
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue (in million dollars)")
plt.legend()
# Add trend line
x = matplotlib.dates.date2num(df.index)
y = df.revenue
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")
plt.show()


# SPLIT TO TRAINING AND TEST
# C4.  Explain the steps you used to prepare the data for analysis, including the training and test set split.

# Split the df into training and testing sets
train_data = df.iloc[:-60]
test_data = df.iloc[-60:]

# C5.  Provide a copy of the cleaned data set.
df.to_csv("teleco_cleaned.csv")
train_data.to_csv("train.csv")
test_data.to_csv("test.csv")

# D1. Analyze the time series data set by doing the following: Report the annotated findings with visualizations of your data analysis, including the following elements: the presence or lack of a seasonal component, trends, the autocorrelation function, the spectral density, the decomposed time series, confirmation of the lack of trends in the residuals of the decomposed series

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram

# Assuming df_diff is your DataFrame with a datetime index and a 'value_diff' column

# Decompose the time series
decomposition = seasonal_decompose(
    df_diff.revenue, model="additive", period=12
)  # Adjust 'period' as necessary

# Plotting the decomposed components
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Observed
axs[0].plot(decomposition.observed)
axs[0].set_title("Observed")
# Trend
axs[1].plot(decomposition.trend)
axs[1].set_title("Trend")
# Seasonal
axs[2].plot(decomposition.seasonal)
axs[2].set_title("Seasonal")
# Residual
axs[3].plot(decomposition.resid)
axs[3].set_title("Residual")

plt.tight_layout()
plt.show()

# Autocorrelation Function (ACF)
plot_acf(df_diff.revenue, lags=50)
plt.title("Autocorrelation Function")
plt.show()

# Spectral Density
frequencies, spectrum = periodogram(df_diff.revenue)
plt.figure(figsize=(10, 4))
plt.plot(frequencies, spectrum)
plt.title("Spectral Density")
plt.xlabel("Frequency")
plt.ylabel("Density")
plt.show()

# D2.  Identify an autoregressive integrated moving average (ARIMA) model that accounts for the observed trend and seasonality of the time series data.
input("Press enter to begin auto arima. This may take a while...")

model = auto_arima(
    train_data,
    start_p=0,
    d=1,
    start_q=0,
    max_p=5,
    max_d=5,
    max_q=5,
    start_P=0,
    D=1,
    start_Q=0,
    max_P=5,
    max_D=5,
    max_Q=5,
    m=12,
    seasonal=True,
    error_action="warn",
    trace=True,
    supress_warnings=True,
    stepwise=True,
    random_state=493,
    n_fits=50,
)

print(model.summary())

input("Press enter to continue...")

# D3.  Perform a forecast using the derived ARIMA model identified in part D2.
model = ARIMA(df.revenue, order=(1, 1, 0), seasonal_order=(5, 1, 0, 12))

results = model.fit()

prediction = pd.DataFrame(results.predict(n_periods=12), index=test_data.index)
prediction.columns = ["revenue"]
print(prediction)

# E1.  Summarize your findings and assumptions by doing the following: Discuss the results of your data analysis, including the following points: the selection of an ARIMA model, the prediction interval of the forecast, a justification of the forecast length, the model evaluation procedure and error metric.

rmse = mean_squared_error(test_data, prediction, squared=False)
print("RMSE:", rmse)

# E2.  Provide an annotated visualization of the forecast of the final model compared to the test set.

plt.figure(figsize=(12, 6))
plt.plot(train_data, label="Training")
plt.plot(test_data, label="Testing")
plt.plot(prediction, label="Predicted")
plt.legend(loc="upper left")
plt.show()

prediction = results.get_prediction(start=len(df), end=len(df) + 365)
plt.figure(figsize=(12, 6))
plt.plot(df, label="Observed")
plt.plot(prediction.predicted_mean, label="Forecast")
plt.legend(loc="upper left")
plt.show()
