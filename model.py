import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("CarPrice_Assignment.csv")

X_train = data[['wheelbase', 'carlength', 'curbweight', 'enginesize', 'boreratio', 'horsepower']]
y_train = data["price"]

forest = RandomForestRegressor()
forest.fit(X_train, y_train)

filename = 'model.sav'
pickle.dump(forest, open(filename, 'wb'))
