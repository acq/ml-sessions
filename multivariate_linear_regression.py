import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
# dataset : http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
# features: age, sex, body mass index, average blood pressure, blood serum measurements 1 -> 6
# target : "quantitative measure of disease progression one year after baseline"

X = diabetes.data
y = diabetes.target

X = np.insert(X, 0, 1, axis=1) # insert a column of "1"s in first position
Xt = np.transpose(X) # or X.T
pinv = np.linalg.pinv(Xt.dot(X))
theta = pinv.dot(Xt).dot(y)

print("theta : " + str(theta))
error = y - X.dot(theta)
print("J(a,b) = " + str(error.dot(error) / (2 * error.shape[0])))

# less manual :
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

scatter = plt.scatter(y, X.dot(theta),  color='black', label='real vs estimated')
line, = plt.plot(y, y, color='blue', linewidth=3, label='identity')
plt.legend(handles=[scatter, line], loc='lower right')
plt.show()
