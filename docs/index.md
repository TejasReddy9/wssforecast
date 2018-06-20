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

*   Now, features are reproduced. Dummies are labelled by creating seerate columns. Don't forget to drop NA entries from both testing and training data. Also, drop those features which exists in training set that doesn't exist in the testing set and vice-versa.
```python
def find_features(train, test, splitset):
        if splitset==True:
            def xdums(df):
                dums = pd.get_dummies(pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.week)
                dums.columns = map(lambda x:"Week_"+str(x), dums.columns.values)
                return dums
        else:
            def xdums(df):
            dums = pd.get_dummies(df["Store"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Store_" + str(x), dums.columns.values)
            res = dums
            dums = pd.get_dummies(df["Dept"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Dept_" + str(x), dums.columns.values)
            res = res.join(dums)
            dums = pd.get_dummies(df["Type"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Type_" + str(x), dums.columns.values)
            res = res.join(dums)
            dums = pd.get_dummies(df["Date"])
            dums = dums.set_index(df.index)
            dums.columns = map(lambda x: "Week_" + str(x), dums.columns.values)
            res = res.join(dums)
            return res
        train_x = xdums(train).join(train[["IsHoliday", "Size", "Year", "Day"]])
        test_x = xdums(test).join(test[["IsHoliday", "Size", "Year", "Day"]])
        train_x = train_x.dropna(axis=1)
        test_x = test_x.dropna(axis=1)
        train_y = train.dropna(axis=1)["Weekly_Sales"]
        for feature in train_x.columns.values:
            if feature not in test_x.columns.values:
                train_x = train_x.drop(feature, axis=1)
        for feature in test_x.columns.values:
            if feature not in train_x.columns.values:
                test_x = test_x.drop(feature, axis=1)
        return train_x, train_y, test_x
```

*   For estimator, I have used Gradient Boosting Regressor available in scikit-learn package. Refer [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html). Try more fiddling with the parameters mentioned in the docs. I've used a loss function which follows least squares regression with least absolute deviation solely based on order information of the input variables.
```python
estimator = GradientBoostingRegressor(loss="huber")
def estimates(train, test, splitset):
        train_x, train_y, test_x = find_features(train, test, splitset)
        estimator.fit(train_x, train_y)
        res = pd.DataFrame(index = test_x.index)
        res["Weekly_Sales"] = estimator.predict(test_x)
        res["Id"] = res.index
        return res
```

*   Actual prediction model is from the subproblems grouped by store_dept. Exceptions are used when I was debugging.
```python
out = pd.DataFrame()
count = 0
for key in testdict.keys():
        count+=1
        try:
            ot = estimates(traindict[key], testdict[key], True)
            out = pd.concat([out, ot])
        except Exception as e:
            print(str(e))
        if count%20==0:
            print("Modelling.... "+ str(100*list(testdict.keys()).index(key)/len(testdict.keys())) +"%")
```

*   Now, let's play with main data. Let's predict on that, if we find any null values(NA), replace them with the result from subproblems.
```python
sout = estimates(train, test, False)
sout = sout.join(out, how="left", lsuffix="_Backup")
sout["Weekly_Sales"] = sout["Weekly_Sales"].fillna(sout["Weekly_Sales_Backup"])
```

*   Finally, output a csv file in the required format.
```python
sout["Id"] = sout["Id"].fillna(sout["Id_Backup"])
sout = sout.drop(["Weekly_Sales_Backup", "Id_Backup"], axis=1)
sout.to_csv("submission.csv", index=False)
```

## Results from predictions
Based on weighted mean absolute error(WMAE), points scored in Kaggle is 11126.41686. Please feel free to download, fork, send pull requests and comment on this project.
