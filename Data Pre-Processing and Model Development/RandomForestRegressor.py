import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

laptop = pd.read_csv('Cleaned_Laptop_Dataset.csv')
laptop.drop(columns=['Unnamed: 0'], inplace=True)
X = laptop.drop(columns=['Price_MYR'])
y = np.log(laptop['Price_MYR'])

#TrainTestSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False), [0, 1, 8, 11, 12])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print('Root Squared Score      :', r2_score(y_test,y_pred))
print('Mean Absolute Error     :', mean_absolute_error(y_test,y_pred))
print('Mean Square Error       :', mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error :', np.sqrt(mean_squared_error(y_test, y_pred)))

pickle.dump(pipe,open('RandomForestRegressorModel.pkl','wb'))