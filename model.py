import pandas as pd 
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime as dt 

# 
# 
# Reading and Merging store-department data into the training and testing datasets
# 
# 
# 
print("Reading and Merging Data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

stores = pd.read_csv("stores.csv")
train = train.merge(stores, how="left", on="Store")
test = test.merge(stores, how="left", on="Store")

# set primary index
train["Id"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str) + "_" + train["Date"].astype(str)
train = train.set_index("Id")
test["Id"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)
test = test.set_index("Id")

# for group by store dept
train["store_dep"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str)
test["store_dep"] = test["Store"].astype(str) + "_" + test["Dept"].astype(str)

# year
train["Year"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.year
test["Year"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.year

# day
train["Day"] = pd.to_datetime(train["Date"], format="%Y-%m-%d").dt.day
test["Day"] = pd.to_datetime(test["Date"], format="%Y-%m-%d").dt.day

# 
# 
# Splitting datasets and grouping by store, dept
# 
# 
# 
print("Splitting the datasets into subsets...")

traindict = {}
testdict = {}
for i in set(test["store_dep"].tolist()):
    traindict[i] = train[train["store_dep"]==i]
    testdict[i] = test[test["store_dep"]==i]


# 
# 
# For each subproblem, this function is run to get features w.r.t the subproblem-dataset
# These subproblems are grouped by store,dept.. above splitted ones
# 
# 
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


# 
# 
# Analyzing various gradient boosting variants, using parameters in scikit documentation
# 
# 
# 
# estimator = GradientBoostingRegressor(loss="ls")
# estimator = GradientBoostingRegressor(loss="lad")
estimator = GradientBoostingRegressor(loss="huber")
# estimator = GradientBoostingRegressor(loss="quantile")


# 
# 
# Fitting and Predicting part using the above estimator
# 
# 
# 
def estimates(train, test, splitset):
    train_x, train_y, test_x = find_features(train, test, splitset)
    estimator.fit(train_x, train_y)
    res = pd.DataFrame(index = test_x.index)
    res["Weekly_Sales"] = estimator.predict(test_x)
    res["Id"] = res.index

    return res

# 
# 
# Solving for each subproblem.. A complete model
# 
# 
# 
print("Beginning main model...")
out = pd.DataFrame()
count = 0
for key in testdict.keys():
    count+=1
    try:
        ot = estimates(traindict[key], testdict[key], True)
        out = pd.concat([out, ot])
        print("..")
    except Exception as e:
        print(str(e))
    if count%20==0:
        print("Modelling.... "+ str(100*list(testdict.keys()).index(key)/len(testdict.keys())) +"%")


# 
# 
# Expanding and making large table and throwing it to get estimated, and a way to fill those missing data
# 
# 
# 
print("Creating giant model to fill in for those pesky missing datas... Probably going to take a while.")
sout = estimates(train, test, False)
sout = sout.join(out, how="left", lsuffix="_Backup")
sout["Weekly_Sales"] = sout["Weekly_Sales"].fillna(sout["Weekly_Sales_Backup"])

# 
# 
#
# Submission format
# 
# 
# 
sout["Id"] = sout["Id"].fillna(sout["Id_Backup"])
sout = sout.drop(["Weekly_Sales_Backup", "Id_Backup"], axis=1)

sout.to_csv("submission.csv", index=False)

# import pandas as pd 
# sub = pd.read_csv("submission.csv")
# sam = pd.read_csv("sampleSubmission.csv")
# print(sam.describe())
# print(sub.describe())