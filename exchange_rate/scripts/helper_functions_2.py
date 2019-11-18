import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
####################################################################################################


def standardizer(X_train, X_test):
    # Instantiate the class
    scaler = StandardScaler()

    # Fit transform the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Only transform the test set
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
####################################################################################################


def mae(y_test, y_pred):
    """Mean absolute error."""
    mae = np.mean(np.abs((y_test - y_pred)))
    return mae


def rmse(y_test, y_pred):
    """Root mean squared error."""
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse


def r_squared(y_test, y_pred):
    """r-squared (coefficient of determination)."""
    mse = np.mean((y_test - y_pred)**2)  # mean squared error
    var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
    r_squared = 1 - mse / var

    return r_squared
####################################################################################################


def Test_prediction(model, n_training_samples, n_training_label,
                    n_test_samples, n_test_label):
    """Test prediction function"""
    model.fit(n_training_samples, n_training_label)
    test_pred = model.predict(n_test_samples)
    return test_pred
####################################################################################################


def diagnostic_plot(y_pred, y_true):
    """Diagnostic plot"""
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    residual = y_pred - y_true
    r2 = round(r_squared(y_true, y_pred), 3)
    rm = round(rmse(y_true, y_pred), 3)

    ax[0].scatter(y_pred, residual, color='b')
    ax[0].set_xlim([0.7, 1])
    ax[0].hlines(y=0, xmin=-0.1, xmax=1, lw=2, color='r')
    ax[0].set_xlabel('Predicted values')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title('Residual plot')

    ax[1].scatter(y_pred, y_true, color='b')
    ax[1].plot([0.7, 1], [0.7, 1], color='r')
    ax[1].set_xlim([0.7, 1])
    ax[1].set_ylim([0.7, 1])
    ax[1].text(.75, .98, r'$R^2 = {},~ RMSE = {}$'.format(
        str(r2), str(rm)), fontsize=20)
    ax[1].set_xlabel('Predicted values')
    ax[1].set_ylabel('Actual values')
    ax[1].set_title('Residual plot')
####################################################################################################
