# Codecool Spectroscopy Analysis
Codecool Spectroscopy Analysis is a sophisticated Python application tailored for advanced data processing. Originating from an advanced module at the Software Development Academy, its core functionality lies in scrutinizing spectroscopy datasets, categorizing them by substance, and subsequently projecting amplitude values via linear and polynomial regression models.

This script came into fruition as a pivotal recruitment qualifier for Coolcode Upskill. The overarching mission was to harness a Machine Learning algorithm to accurately predict the spectroscopic signal outputs of sensors for isopropanol and water, all while leveraging measured metrics in the air as the foundation. Data extraction spanned across an array of files, amalgamating measurements from 10 distinct sensors corresponding to air, water, and isopropanol.

Upon meticulous validation of data integrity, the dataset was strategically partitioned into training and testing segments. The results for the Linear Regression during training were as follows:

**Training - Linear Regression:**
- R2: 0.87704
- R2 Adjusted: 0.87702
- MSE: 5.67981
- RMSE: 2.38324
- MAE: 1.90228
- MAPE: 135.65054
- sMAPE: 47.31598
- AIC: 219001.72665
- BIC: 219025.98508

**Test - Linear Regression:**
- R2: 0.87992
- R2 Adjusted: 0.87986
- MSE: 5.50579
- RMSE: 2.34644
- MAE: 1.87831
- MAPE: 208.40155
- sMAPE: 46.19237
- AIC: 54388.95469
- BIC: 54409.05423

**Test - Polynomial Regression:**
- R2: 0.96691
- R2 Adjusted: 0.96689
- MSE: 1.52321
- RMSE: 1.23418
- MAE: 0.97769
- MAPE: 76.27707
- sMAPE: 26.54695
- AIC: 38801.09511

Despite the robustness of air as a predictor in the linear model, certain anomalies were discerned in the outcomes. This necessitates further exploration into non-linear regression paradigms to unearth potential avenues for model enhancement. The Polynomial Regression showed a significant improvement, especially in the R2 value, indicating a stronger fit to the data.

## About the Application
The SDA Spectroscopy Analysis application was developed as a part of an advanced course offered by the Software Development Academy. It leverages libraries such as Pandas, Numpy, and Scikit-Learn to preprocess spectroscopy datasets, train regression models, and visualize results.

## Features
1. **Data Loading**: Loads and merges datasets from a directory.
2. **Linear Regression**: Trains a linear regression model on the loaded data and provides detailed performance metrics.
3. **Polynomial Regression**: Uses a second-degree polynomial regression for a more nuanced modeling approach.
4. **Visualization**: Plots the true vs. predicted amplitude values for better interpretation.

## Running the Application
1. Clone the repository.
2. Ensure that you have the required libraries installed (`pandas`, `numpy`, `sklearn`, `matplotlib`).
3. Navigate to the project's root directory.
4. Run the main script to execute the application.

## Contributing
Contributions are always welcome! If you're interested in enhancing the functionality or improving the modeling techniques, please fork the repository, make your modifications, and then submit a pull request.

## License
This project is open-sourced under the MIT License.
