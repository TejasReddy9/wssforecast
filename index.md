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
