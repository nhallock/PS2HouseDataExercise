# House Price Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn

# import data

trainData = pd.read_csv('train.csv')
#testData = pd.read_csv("test.csv")

# Take in all data
train = trainData.iloc[0:1000,:]
test  = trainData.iloc[1001:1459,:]

# Explore the data
#print(train.head())
#print(test.head())

trainSalePrice = train["SalePrice"]
testSalePrice = test["SalePrice"]

#print(salePrice)
#print(salePrice.describe())

# Plot the Data
# f1 = plt.figure()
# plt.figure(f1.number)
# plt.hist(salePrice)
# plt.ylabel("Sale Price")
# plt.xlabel("Bin Number")
# plt.title("Histogram")
# plt.show()

# Select numeric columns
# calculate correlation factor
trainNumeric = train.select_dtypes(include=[np.number])
testNumeric = test.select_dtypes(include=[np.number])
#print("Numeric: \n", numeric.shape)

trainCorr = trainNumeric.corr()
cols = trainCorr["SalePrice"].sort_values(ascending=False)[0:3].index
#print("Corr: ", cols)

# Pick out X and Y values
trainX = train[cols]
trainY = train["SalePrice"]
trainX = trainX.drop(["SalePrice"], axis=1)

testX = test[cols]
testY = test["SalePrice"]
testX = testX.drop(["SalePrice"], axis=1)


#print("X: ", X)

lr = linear_model.LinearRegression()
model = lr.fit(trainX,trainY)
trainPredictions = model.predict(trainX)

testPredictions = model.predict(testX)

# How good is the model
print("R^2: ", model.score(testX,testY))
sklearn.metrics.mean_squared_error(testPredictions, testY)

# Scatter Plot of Predictions
#plt.scatter(predictions, Y)
#plt.show()




