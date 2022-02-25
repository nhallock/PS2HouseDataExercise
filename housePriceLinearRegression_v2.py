# House Price Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
import csv

# import data

trainData = pd.read_csv('train.csv')
testData = pd.read_csv("test.csv")

# Take in all data
train = trainData.iloc[:,:]
test  = trainData.iloc[:,:]

# Explore the data
#print(train.head())
#print(test.head())

trainSalePrice = train["SalePrice"]

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

corr = trainNumeric.corr()
cols = corr["SalePrice"].sort_values(ascending=False)[0:3].index
#print("Corr: ", cols)

# Pick out X and Y values
trainX = train[cols]
trainY = train["SalePrice"]
trainX = trainX.drop(["SalePrice"], axis=1)

testX = test[cols[1:3]]
testY = test['SalePrice']

# Create model
lr = linear_model.LinearRegression()
model = lr.fit(trainX,trainY)
predictions = model.predict(testX)

# How good is the model
print("R^2: ", model.score(testX, testY))
print(sklearn.metrics.mean_squared_error(predictions, testY))
print(predictions)

# Create list of all IDs for printing to csv
ids = testNumeric["Id"]
print("Ids: ", ids)


# Write Predictions to .csv
dict = {'ID': ids, "Prediction": predictions}
df = pd.DataFrame(dict)
df.to_csv('predictions.csv', index=False)








