import pandas as pd
import numpy as np
import seaborn as sns
import json
from pandas import json_normalize
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic, ccf
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit


import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from missingpy import MissForest
from missingpy import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from functools import reduce
from sklearn.model_selection import KFold
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

def set_target_in_ph(data, prediction_horizon):
    ph_col_name = 'target_in_oh_{}'.format(prediction_horizon)
    data[ph_col_name] = data['sales_quantity'].shift(-1 * prediction_horizon)
    return data

def check_stationarity(timeseries_p, col_name):
    print('Results of Dickey-Fuller Test for {}:'.format(col_name))
    dftest = adfuller(timeseries_p, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def detrend(series, window_size=10):
    moving_avg = series.rolling(window=window_size).mean()
    series = series - moving_avg
    return series


def difference(series):
    series = series - series.shift()
    return series


def set_vector_date(data, vector_length, numeric_cols):
    vector_columns = numeric_cols
    for col in vector_columns:
        col_name = col
        for i in range(vector_length):
            new_col = col_name + '_t-{}'.format(i + 1)
            data[new_col] = data[col].shift(i + 1)
    return data


df = pd.read_csv('../data.csv')

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df = df.sort_values('day')
df.drop(['revenue'], axis=1, inplace=True)
df['day'] = pd.to_datetime(df['day'])

df = df.drop_duplicates()
df['unit_profit'] = df['suggested_retail_price'] - df['purchase_price']
df['dayofweek'] = df['day'].dt.dayofweek
df['dayofmonth'] = df['day'].dt.day
df['month'] = df['day'].dt.month


df_pred=df[['day','purchase_price','suggested_retail_price','unit_profit','orders_quantity','sales_quantity']]

df_pred['purchase_price'] = detrend(df_pred['purchase_price'])
df_pred['purchase_price'] = difference(df_pred['purchase_price'])

df_pred['suggested_retail_price'] = detrend(df_pred['suggested_retail_price'])
df_pred['suggested_retail_price'] = difference(df_pred['suggested_retail_price'])

df_pred['unit_profit'] = detrend(df_pred['unit_profit'])
df_pred['unit_profit'] = difference(df_pred['unit_profit'])

df_pred = df_pred.dropna(axis=0)

prediction_horizon = 1
vector_length = 6
numeric_cols=['purchase_price','suggested_retail_price','unit_profit','orders_quantity','sales_quantity']

df_pred = set_vector_date(df_pred, vector_length, numeric_cols)
df_pred = set_target_in_ph(df_pred, prediction_horizon)
df_pred.dropna(inplace=True)
df_pred.head()

x = df_pred.drop(['day', 'target_in_oh_1'], axis = 1)
y = df_pred['target_in_oh_1']

model = pickle.load(open('gbm.json', 'rb'))

y_pred = model.predict(x)

print('result is:', int(y_pred[len(y_pred)-1]))

