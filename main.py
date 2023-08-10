import os
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Constants
DIR_PATH = "./Dataset"


def load_data_from_directory(directory_path):
    """Load datasets from the directory."""
    dfs = defaultdict(list)

    for filepath in glob(os.path.join(directory_path, "*/*")):
        temp_df = pd.read_csv(filepath, names=['wavelength', 'amplitude'])
        filename = os.path.basename(filepath)

        try:
            base_name = filename.rsplit('.', 1)[0]
            parts = base_name.split('_', 1)
            if len(parts) != 2:
                raise ValueError

            sensor, substance = parts
            sensor = int(sensor[-2:])
        except ValueError:
            print(f"Problem with filename: {filename}")
            continue

        temp_df['sensor'] = sensor
        dfs[substance].append(temp_df)

    return dfs


def merge_dfs(dfs, substances):
    """Merge dataframes by substance."""
    merged_df = pd.concat(dfs[substances[0]])
    merged_df.reset_index(drop=True, inplace=True)
    merged_df.rename(columns={"amplitude": f"{substances[0]}_amplitude"}, inplace=True)

    for substance in substances[1:]:
        temp_df = pd.concat(dfs[substance])
        temp_df.reset_index(drop=True, inplace=True)
        temp_df.rename(columns={"amplitude": f"{substance}_amplitude"}, inplace=True)
        merged_df = pd.merge(merged_df, temp_df, on=["wavelength", "sensor"])

    return merged_df


def train_linear_regression(X_train, y_train):
    """Train linear regression model."""
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg


def show_performance_stats(y_true, predictions, X):
    """Print performance statistics."""
    n, k = len(y_true), X.shape[1]

    r2 = metrics.r2_score(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, predictions)
    mape_series = 100 * np.abs((y_true - predictions) / y_true)
    mape = mape_series.mean(axis=0).mean()
    smape_series = 100 * 2 * np.abs(predictions - y_true) / (np.abs(predictions) + np.abs(y_true))
    smape = smape_series.mean(axis=0).mean()
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Compute log-likelihood
    residuals = y_true - predictions
    sse = np.sum(residuals ** 2)
    sigma2 = sse / n
    log_likelihood = (-n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma2) - sse / (2 * sigma2)).sum()

    # Compute AIC and BIC
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    print("MODEL PERFORMANCE STATS")
    print(f"R2: {r2:.5f}")
    print(f"R2 Adjusted: {r2_adj:.5f}")
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"MAPE: {mape:.5f}")
    print(f"sMAPE: {smape:.5f}")
    print(f"AIC: {aic:.5f}")
    print(f"BIC: {bic:.5f}")


def plot_true_vs_predicted(X_test, y_test, test_predictions):
    """Plot true vs. predicted data."""
    plt.scatter(X_test['wavelength'], pd.DataFrame(test_predictions)[0], label="Water predictions", s=1)
    plt.scatter(X_test['wavelength'], y_test['water_amplitude'], label="Water amplitude", s=1)
    plt.legend()
    plt.show()

    plt.scatter(X_test['wavelength'], pd.DataFrame(test_predictions)[1], label="Izopropanol predictions", s=1)
    plt.scatter(X_test['wavelength'], y_test['izopropanol_amplitude'], label="Izopropanol amplitude", s=1)
    plt.legend()
    plt.show()


def main():
    # Load data
    dfs = load_data_from_directory(DIR_PATH)
    df = merge_dfs(dfs, ["air", "water", "izopropanol"])

    X = df[['wavelength', 'air_amplitude', 'sensor']]
    y = df[['water_amplitude', 'izopropanol_amplitude']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression
    lin_reg = train_linear_regression(X_train, y_train)
    train_predictions = lin_reg.predict(X_train)
    test_predictions = lin_reg.predict(X_test)

    # Show performance stats
    print("TRAINING - Linear Regression")
    show_performance_stats(y_train, train_predictions, X_train)
    print("\nTEST - Linear Regression")
    show_performance_stats(y_test, test_predictions, X_test)

    # Plot results
    plot_true_vs_predicted(X_test, y_test, test_predictions)

    # Train polynomial regression
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X_train, y_train)
    test_poly_predictions = poly_model.predict(X_test)

    # Show performance stats for polynomial regression
    print("\nTEST - Polynomial Regression")
    show_performance_stats(y_test, test_poly_predictions, X_test)


if __name__ == "__main__":
    main()
