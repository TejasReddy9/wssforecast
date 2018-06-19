---
layout: default
---

# Overview
Historical Sales Data of 45 Walmart stores each containing a number of departments is provided. I have made a regression model  which predicts the department-wide sales for each store.

## Requirements
Install Python3.x, and install these dependencies using pip - scikit-learn, sklearn, scipy, pandas.
```
pip3 install dependency_name
```

## Data
As data shouldn't be posted publicly, please refer this [link](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data).
1.  Stores. (Store, Type, Size)
2.  Features. These are additional features used which have less direct impact. Markdowns mentioned are special holidays. (Store, Date, Temperature, Fuel_Price, MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI, Unemployment, IsHoliday)
3.  Training Dataset. (Store, Dept, Date, Weekly_Sales, IsHoliday)
4.  Testing Dataset. (Store, Dept, Date, IsHoliday)
5.  Sample Submission Format.

## Approach
*   First, let's read-in data and modify the training and testing data by merging stores data into it. Assuming data is downloaded in the same directory.
```python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
stores = pd.read_csv("stores.csv")
train = train.merge(stores, how="left", on="Store")
test = test.merge(stores, how="left", on="Store")
```
*   A little cleaning, setting index values of the table for the format specified in the sample submission, and added a new column for grouping of the data w.r.t store and department.
```python
train["Id"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str) + "_" + train["Date"].astype(str)
train = train.set_index("Id")
test["Id"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)
test = test.set_index("Id")
train["store_dep"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str)
test["store_dep"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str)
```
*   Whole problem is divided into subproblems grouped by each department of each store, stored in dictionaries for testing and training data, where keys are the store_dept.
```python
traindict = {}
testdict = {}
for i in set(test["store_dep"].tolist()):
    traindict[i] = train[train["store_dep"]==i]
    testdict[i] = test[test["store_dep"]==i]
```
*   
