import quandl


mydata = quandl.get("YAHOO/INDEX_DJI", start_date="2005-12-01", end_date="2005-12-05")




import pandas as pd


authtoken = 'DrxQ6jniVGwDnrDrrb_Y'

def get_data_quandl(symbol, start_date, end_date):
    data = quandl.get(symbol, start_date=start_date, end_date=end_date, authtoken=authtoken)
    return data


def generate_features(df):
    """ Generate features for a stock/index based on historical price and performance
    Args:
        df (dataframe with columns "Open", "Close", "High", "Low", "Volume", "Adjusted Close")
    Returns:
        dataframe, data set with new features
    """
    df_new = pd.DataFrame()
    # 6 original features
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)
    # 31 original features
    # average price
    df_new['avg_price_5'] = pd.rolling_mean(df['Close'], window=5).shift(1)
    df_new['avg_price_30'] = pd.rolling_mean(df['Close'], window=21).shift(1)
    df_new['avg_price_365'] = pd.rolling_mean(df['Close'], window=252).shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']
    # average volume
    df_new['avg_volume_5'] = pd.rolling_mean(df['Volume'], window=5).shift(1)
    df_new['avg_volume_30'] = pd.rolling_mean(df['Volume'], window=21).shift(1)
    df_new['avg_volume_365'] = pd.rolling_mean(df['Volume'], window=252).shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']
    # standard deviation of prices
    df_new['std_price_5'] = pd.rolling_std(df['Close'], window=5).shift(1)
    df_new['std_price_30'] = pd.rolling_std(df['Close'], window=21).shift(1)
    df_new['std_price_365'] = pd.rolling_std(df['Close'], window=252).shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']
    # standard deviation of volumes
    df_new['std_volume_5'] = pd.rolling_std(df['Volume'], window=5).shift(1)
    df_new['std_volume_30'] = pd.rolling_std(df['Volume'], window=21).shift(1)
    df_new['std_volume_365'] = pd.rolling_std(df['Volume'], window=252).shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']
    # # return
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = pd.rolling_mean(df_new['return_1'], window=5)
    df_new['moving_avg_30'] = pd.rolling_mean(df_new['return_1'], window=21)
    df_new['moving_avg_365'] = pd.rolling_mean(df_new['return_1'], window=252)
    # the target
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


symbol = 'YAHOO/INDEX_DJI'
start = '2001-01-01'
end = '2014-12-31'
data_raw = get_data_quandl(symbol, start, end)
data = generate_features(data_raw)
data.round(decimals=3).head(3)


symbol = 'YAHOO/INDEX_DJI'
start = '1988-01-01'
end = '2015-12-31'
data_raw = get_data_quandl(symbol, start, end)
data = generate_features(data_raw)

# next day prediction
import datetime
start_train = datetime.datetime(1988, 1, 1, 0, 0)
end_train = datetime.datetime(2014, 12, 31, 0, 0)

data_train = data.ix[start_train:end_train]
X_columns = list(data.drop(['close'], axis=1).columns)
y_column = 'close'
X_train = data_train[X_columns]
y_train = data_train[y_column]

start_test = datetime.datetime(2015, 1, 1, 0, 0)
end_test = datetime.datetime(2015, 12, 31, 0, 0)
data_test = data.ix[start_test:end_test]
X_test = data_test[X_columns]
y_test = data_test[y_column]


from sklearn.model_selection import GridSearchCV
# First experiment with linear regression

# SGD is very sensitive to data with features at different scales. Hence we need to do feature scaling before training.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled_train = scaler.transform(X_train)
X_scaled_test = scaler.transform(X_test)

param_grid = {
    "alpha": [1e-5, 3e-5, 1e-4],
    "eta0": [0.01, 0.03, 0.1],
}

from sklearn.linear_model import SGDRegressor
lr = SGDRegressor(penalty='l2', n_iter=1000)
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)

lr_best = grid_search.best_estimator_
# print(grid_search.best_score_)

predictions_lr = lr_best.predict(X_scaled_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print('MSE: {0:.3f}'.format(mean_squared_error(y_test, predictions_lr)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, predictions_lr)))
print('R^2: {0:.3f}'.format(r2_score(y_test, predictions_lr)))



# Next experiment with random forest

param_grid = {
    "max_depth": [30, 50],
    "min_samples_split": [5, 10, 20],

}

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=1000)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# print(grid_search.best_score_)

rf_best = grid_search.best_estimator_
predictions_rf = rf_best.predict(X_test)

print('MSE: {0:.3f}'.format(mean_squared_error(y_test, predictions_rf)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, predictions_rf)))
print('R^2: {0:.3f}'.format(r2_score(y_test, predictions_rf)))




# Finally experiment with SVR
param_grid = {
              "C": [1000, 3000, 10000],
              "epsilon": [0.00001, 0.00003, 0.0001],
              }

from sklearn.svm import SVR
svr = SVR(kernel='linear')
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_scaled_train, y_train)

print(grid_search.best_params_)

svr_best = grid_search.best_estimator_
# print grid_search.best_score_

predictions_svr = svr_best.predict(X_scaled_test)

print('MSE: {0:.3f}'.format(mean_squared_error(y_test, predictions_svr)))
print('MAE: {0:.3f}'.format(mean_absolute_error(y_test, predictions_svr)))
print('R^2: {0:.3f}'.format(r2_score(y_test, predictions_svr)))




import matplotlib.pyplot as plt

dates = data_test.index.values
plot_truth, = plt.plot(dates, y_test, 'k')
plot_lr, = plt.plot(dates, predictions_lr, 'r')
plot_rf, = plt.plot(dates, predictions_rf, 'b')
plot_svr, = plt.plot(dates, predictions_svr, 'g')
plt.legend([plot_truth, plot_lr, plot_rf, plot_svr], ['Truth', 'Linear regression', 'Random forest', 'SVR'])
plt.title('Stock price prediction vs truth')
plt.show()
