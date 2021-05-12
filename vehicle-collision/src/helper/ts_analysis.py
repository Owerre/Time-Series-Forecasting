
# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# stat models
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class TimeSeriesAnalysis:
    """
    A class for analyzing time series data
    """
    def __init__(self) -> None:
        pass

    def plot_timeseries(self, ts, color=None, marker=None, title = ''):
        """
        Plot univariant time series data
        """
        ts.plot(marker=marker, color=color, figsize=(18,8))
        plt.xlabel('Year', fontsize=20)
        plt.ylabel('Monthly fatality rate', fontsize=20)
        plt.title(title, fontsize=20)

    def plot_ts(self, ts, title = '', nlags=None):
        """
        This function plots the original time series together rolling mean 
        and standard deviations and its ACF and partial ACF. 
        It also performs the Dickey-Fuller test
        """
        gridsize = (2, 2)
        ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2)
        ax2 = plt.subplot2grid(gridsize, (1, 0))
        ax3 = plt.subplot2grid(gridsize, (1, 1))

        # Rolling statistic
        rolling_mean = ts.rolling(window=24).mean()
        rolling_std = ts.rolling(window=24).std()

        # Plot original time series and rolling mean & std
        ts.plot(color='r', marker='o', ax=ax1, label='Original')
        rolling_mean.plot(color='b', ax=ax1, label='Rolling mean')
        rolling_std.plot(color='g', ax=ax1, label='Rolling Std')
        ax1.set_xlabel('Year', fontsize=20)
        ax1.set_ylabel('Monthly fatality rate', fontsize=20)
        ax1.set_title(title, fontsize=20)
        ax1.legend(loc='best')

        # Plot ACF
        plot_acf(ts, lags=nlags, ax=ax2)
        ax2.set_xlabel('Lag', fontsize=20)
        ax2.set_ylabel('ACF', fontsize=20)
        ax2.set_title('Autocorrelation', fontsize=20)

        # Plot PACF
        plot_pacf(ts, lags=nlags, ax=ax3)
        ax3.set_xlabel('Lag', fontsize=20)
        ax3.set_ylabel('PACF', fontsize=20)
        ax3.set_title('Partial Autocorrelation', fontsize=20)

        # Perform Dickey-Fuller test
        adf_results = adfuller(ts.values)
        print('Test statistic:', adf_results[0])
        print('p-value:', adf_results[1])
        for key, value in adf_results[4].items():
            print('Critial Values (%s): %0.6f' % (key, value))

    def diagnostic_plot(self, y_true, y_pred):
        """
        Diagnostic Plot of Residuals for the Out-of-Sample Forecast
        """
        gridsize = (1, 2)
        ax1 = plt.subplot2grid(gridsize, (0, 0))
        ax2 = plt.subplot2grid(gridsize, (0, 1))
    #     ax3 = plt.subplot2grid(gridsize, (1, 0))
    #     ax4 = plt.subplot2grid(gridsize, (1, 1))

        residual = y_true-y_pred   # compute the residual

        ax1.scatter(y_pred, residual)
        ax1.set_xlim([min(y_true) - 0.02, max(y_true) + 0.02])
        ax1.axhline(y=0, lw=2, color='k')
        ax1.set_xlabel('Predicted value', fontsize=20)
        ax1.set_ylabel('Residual', fontsize=20)
        ax1.set_title('Residual plot', fontsize=20)

        ax2.scatter(y_pred, y_true)
        ax2.plot([min(y_true) - 0.02, max(y_true) + 0.02],
                [min(y_true) - 0.02, max(y_true) + 0.02],
                color='k')
        ax2.set_xlim([min(y_true) - 0.02, max(y_true) + 0.02])
        ax2.set_ylim([min(y_true) - 0.02, max(y_true) + 0.02])
        ax2.set_xlabel('Predicted value', fontsize=20)
        ax2.set_ylabel('Actual value', fontsize=20)
        ax2.set_title('Residual plot', fontsize=20)

    #     ax3.plot(acorr_ljungbox(residual, lags = nlags)[1],'o')
    #     ax3.axhline(y=0.05,linestyle= '--', color = 'k')
    #     ax3.set_xlabel('Lag', fontsize = 20)
    #     ax3.set_ylabel('p-value', fontsize = 20)
    #     ax3.set_title('p-values for Ljung-Box statistic', fontsize = 20)

    #     ax4.scatter(data.index.year, residual)
    #     ax4.set_xlabel('Year', fontsize = 20)
    #     ax4.set_title('Residuals vs. years', fontsize = 20)
    #     ax4.set_ylabel('Residual', fontsize = 20)
    #     ax4.set_xticks([2018, 2019])


    def mape(self, y_true, y_pred):
        """
        Mean absolute percentage error.
        """
        mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
        return mape

    def mae(self, y_true, y_pred):
        """
        Mean absolute error.
        """
        mae = np.mean(np.abs(y_true - y_pred))
        return mae

    def rmse(self, y_true, y_pred):
        """
        Root mean squared error.
        """
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        return rmse

    def r_squared(self, y_true, y_pred):
        """
        r squared (coefficient of determination).
        """
        mse = np.mean((y_true - y_pred)**2)  # mean squared error
        var = np.mean((y_true - np.mean(y_true))**2)  # sample variance
        r_squared = 1 - mse / var
        return r_squared

