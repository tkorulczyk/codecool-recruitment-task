from glob import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import seaborn as sns

# 1. Import all data to df set
DIR_PATH = "./Dataset"
filepath_list = glob(DIR_PATH+"/*/*")

filepath_list

dfs = defaultdict(list)

# 2. Extract info about sensor from file name
for filepath in filepath_list:
    temp_df = pd.read_csv(filepath, names=['wavelength', 'amplitude'])
    filename = filepath.split("/")[-1]
    substance = filename.split(".")[0].split("_")[-1]
    sensor = filename.split(".")[0].split("_")[1][-2:]

    sensor2 = sensor[1][-2:]
    temp_df['sensor'] = int(sensor)

    dfs[substance].append(temp_df)

df_air = pd.concat(dfs["air"])
df_water = pd.concat(dfs["water"])
df_izopropanol = pd.concat(dfs["izopropanol"])

df_air.reset_index(drop=True, inplace=True)
df_water.reset_index(drop=True, inplace=True)
df_izopropanol.reset_index(drop=True, inplace=True)

df_air.rename(columns={"amplitude": "air_amplitude"}, inplace=True)
df_water.rename(columns={"amplitude": "water_amplitude"}, inplace=True)
df_izopropanol.rename(columns={"amplitude": "izopropanol_amplitude"}, inplace=True)

df = pd.merge(df_air, df_water, on=["wavelength", "sensor"])
df = pd.merge(df, df_izopropanol, on=["wavelength", "sensor"])


# 3. Train the model
X = df_air
y = df[['water_amplitude', 'izopropanol_amplitude']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train.iloc[0]['wavelength']
X_train[X_train['wavelength']==X_train.iloc[1]['wavelength']].sort_values('sensor')
X_test[X_test['wavelength']==X_train.iloc[1]['wavelength']].sort_values('sensor')

# 4. Use linear regresison to predict
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


#################################################################
import statsmodels.api as sm

# fit the regression model

# print(df_air['air_amplitude'])
# print(df_water['water_amplitude'])

# X_train = X_train.values.reshape(-1,1)
# df_air = df_air.values.reshape(-1,1)
#
# reg = sm.OLS(df_air['air_amplitude'], X_train).fit()
# print(reg.summary())

pd.DataFrame(zip(X_train.columns, lin_reg.coef_[0]))
pd.DataFrame(zip(X_train.columns, lin_reg.coef_[1]))

train_predictions = lin_reg.predict(X_train)
test_predictions = lin_reg.predict(X_test)


# 5. Show performance statistics
import sklearn.metrics as metrics

print("MODEL PERFORMACNE STATS")
print("TRAINING")
print(f"R2: {metrics.r2_score(y_train, train_predictions)}")
print(f"MSE: {metrics.mean_squared_error(y_train, train_predictions)}")

print("TEST")
print(f"R2: {metrics.r2_score(y_test, test_predictions)}")
print(f"MSE: {metrics.mean_squared_error(y_test, test_predictions)}")

from statsmodels.stats.stattools import durbin_watson

#perform Durbin-Watson test


# 6. Show plots of true and predicted data
#plt.rcParams["figure.figsize"] = (20,10)
plt.scatter(X_test['wavelength'], pd.DataFrame(test_predictions)[0], label=f"Water predictions", s=1)
plt.scatter(X_test['wavelength'], y_test['water_amplitude'], label=f"Water amplitude", s=1)
plt.legend()
plt.show()

plt.scatter(X_test['wavelength'], pd.DataFrame(test_predictions)[1], label=f"Izopropanol predictions", s=1)
plt.scatter(X_test['wavelength'], y_test['izopropanol_amplitude'], label=f"Izopropanol amplitude", s=1)
plt.legend()
plt.show()

# 7. Test assumptions of regression
# Check for Linearity
f = plt.figure(figsize=(14, 5))
ax = f.add_subplot(121)
sns.scatterplot(x=y_test['water_amplitude'], y=pd.DataFrame(test_predictions)[0], ax=ax, color="r")
ax.set_title("Test liniowej zależności:\n Actual Vs Predicted value")

# Check for Residual normality & mean
ax = f.add_subplot(122)
sns.distplot((y_test['water_amplitude'] - pd.DataFrame(test_predictions)[0]), ax=ax, color="b")
ax.axvline((y_test['water_amplitude'] - pd.DataFrame(test_predictions)[0]).mean(), color="k", linestyle="--")
ax.set_title("Test dla normalości reszt (residuals): \n Residual eror")


