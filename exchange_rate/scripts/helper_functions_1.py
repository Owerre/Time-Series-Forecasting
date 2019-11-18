# Matplotlib  for data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

# Seaborn for data visualization
import seaborn as sns
# Set font scale and style
sns.set(font_scale=2)
sns.set_style('ticks')
plt.style.use('seaborn-white')
mpl.rcParams['font.family'] = 'serif'


def plot_timeseries(timeseries, color=None, marker=None, title=''):
    """This function plots the time series"""

    # Set font size and background color
    sns.set(font_scale=2)
    plt.style.use('ggplot')

    timeseries.plot(marker=marker, color=color, figsize=(15, 6))
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Euro to USD exchange', fontsize=20)
    plt.title(title, fontsize=20)
    plt.grid(True)


def plot_ts(timeseries, title='', nlags=None):
    """This function plots the original time series together rolling mean
    and standard deviations
    and its ACF and partial ACF. It also performs the Dickey-Fuller test
    """

    # Set font size and background color
    sns.set(font_scale=1.5)
    plt.style.use('ggplot')

    gridsize = (2, 2)
    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(gridsize, (1, 0))
    ax3 = plt.subplot2grid(gridsize, (1, 1))

    # Rolling statistic
    rolling_mean = timeseries.rolling(window=24).mean()
    rolling_std = timeseries.rolling(window=24).std()

    # Plot original time series and rolling mean & std
    timeseries.plot(color='r', marker='o', ax=ax1, label='Original')
    rolling_mean.plot(color='b', ax=ax1, label='Rolling mean')
    rolling_std.plot(color='g', ax=ax1, label='Rolling Std')
    ax1.set_xlabel('Year', fontsize=20)
    ax1.set_ylabel('Number of fatalities', fontsize=20)
    ax1.set_title(title, fontsize=20)
    ax1.grid(True)
    ax1.legend(loc='best')

    # Plot ACF
    plot_acf(timeseries, lags=nlags, ax=ax2)
    ax2.set_xlabel('Lag', fontsize=20)
    ax2.set_ylabel('ACF', fontsize=20)
    ax2.set_title('Autocorrelation', fontsize=20)
    ax2.grid(True)

    # Plot PACF
    plot_pacf(timeseries, lags=nlags, ax=ax3)
    ax3.set_xlabel('Lag', fontsize=20)
    ax3.set_ylabel('PACF', fontsize=20)
    ax3.set_title('Partial Autocorrelation', fontsize=20)
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

    # Perform Dickey-Fuller test
    adf_results = adfuller(timeseries.values)
    print('Test statistic:', adf_results[0])
    print('p-value:', adf_results[1])
    for key, value in adf_results[4].items():
        print('Critial Values (%s): %0.6f' % (key, value))


def mae(y_test, y_pred):
    """Mean absolute error."""

    mae = np.abs(y_test - y_pred).mean()
    return mae


def rmse(y_test, y_pred):
    """Root mean squared error."""

    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse


def r_squared(y_test, y_pred):
    """r squared (coefficient of determination)."""

    mse = np.mean((y_test - y_pred)**2)  # mean squared error
    var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
    r_squared = 1 - mse / var

    return r_squared


def Diagnostic_Plot(data, y_true, y_pred, nlags):
    """
    Diagnostic Plot of Residuals for the Out-of-Sample Forecast
    """
    sns.set(font_scale=1.5)
    plt.style.use('ggplot')

    gridsize = (1, 2)
    fig = plt.figure(figsize=(20, 6))
    ax1 = plt.subplot2grid(gridsize, (0, 0))
    ax2 = plt.subplot2grid(gridsize, (0, 1))
#     ax3 = plt.subplot2grid(gridsize, (1, 0))
#     ax4 = plt.subplot2grid(gridsize, (1, 1))

    residual = y_pred - y_true  # compute the residual

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
