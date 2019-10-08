import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# read the data set for red wine
data = pd.read_csv("winequality-red.csv", sep=";")

# get best data subset
'''
data = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"]]
'''

data = data[["fixed acidity", "volatile acidity", "chlorides", "sulphates", "alcohol", "quality"]]

# get prediction value
predict = "quality"

# separate data into inputs and outputs
X = np.array(data.drop(predict, 1))
y = np.array(data[predict])

# create model
linear = linear_model.LinearRegression()

# calculate average accuracy
ave_acc = 0
for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)
    linear.fit(x_train, y_train)
    ave_acc += linear.score(x_test, y_test)

print(f"average accuracy = {ave_acc / 100}")
print("The end")