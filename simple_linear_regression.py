import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

Xt = np.transpose(X) # or X.T
pinv = np.linalg.pinv(Xt.dot(X))
b = pinv.dot(Xt).dot(y)

a = np.mean(y) - b.dot(np.mean(X))

plt.scatter(X, y,  color='black')
plt.plot(X, a + X.dot(b), color='blue', linewidth=3)
plt.show()