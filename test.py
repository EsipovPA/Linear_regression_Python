import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


# read the full data set
data = pd.read_csv("student-mat.csv", sep = ";")

# get required data from full data set
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

"""
best_acc = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best_acc:
        best_acc = acc
        # save the best model
        print("saved model to file")
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# Load the best model achieved
pickle_in = open("student_model.pickle", "rb")

linear = pickle.load(pickle_in)
print(f"loaded model acc1 = {linear.score(x_test, y_test)}")

p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()

