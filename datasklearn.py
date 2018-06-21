from sklearn import datasets,linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
diabetes=datasets.load_diabetes()

diabetes_X = diabetes.data[:, np.newaxis,2]

#split the data into traning and testing sets

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#split the data into traning and testing sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

#create linear regration object

reg = linear_model.LinearRegression()

# Train the model using the training sets
reg.fit(diabetes_X_train,diabetes_Y_train)

# Make predictions using the testing set
diabetes_Y_pred = reg.predict(diabetes_X_test)

#The cofficients
print('Coefficients: \n', reg.coef_)
#mean square error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_Y_test, diabetes_Y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_Y_test, diabetes_Y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_Y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_Y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
