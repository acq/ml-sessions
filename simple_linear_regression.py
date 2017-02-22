import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
# dataset : http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
# features: age, sex, body mass index, average blood pressure, blood serum measurements 1 -> 6
# target : "quantitative measure of disease progression one year after baseline"

X = diabetes.data[:, np.newaxis, 2] # keep only one feature
y = diabetes.target

Xt = np.transpose(X) # or X.T
pinv = np.linalg.pinv(Xt.dot(X))
b = pinv.dot(Xt).dot(y)

a = np.mean(y) - b.dot(np.mean(X))

print("a : " + str(a[0]) + " ; b : " + str(b[0]))
error = y - a - X.dot(b)
print("J(a,b) = " + str(error.dot(error) / (2 * error.shape[0])))

plt.scatter(X, y,  color='black')
plt.plot(X, a + X.dot(b), color='blue', linewidth=3)
plt.show()
