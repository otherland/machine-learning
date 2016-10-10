import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression



df = pd.read_pickle('GOOG.dmp')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]

# volatility: high low percent
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'] / df['Adj. Low'] * 100.0)
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'] / df['Adj. Open'] * 100.0)

df = df[['Adj. Close', 'Adj. Volume', 'HL_PCT', 'PCT_change']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# df without label column
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)

# just label column
y = np.array(df['label'])

# split data for training and testing
# 20% of data used for training
# shuffle data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# train data
Classifier = LinearRegression()
Classifier.fit(X_train, y_train)

# test
accuracy = Classifier.score(X_test, y_test)

print(accuracy)