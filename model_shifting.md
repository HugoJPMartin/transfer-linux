# Linux Transfer Learning notebook - Model Shifting

This notebook aims to show how to do model shifting.

The principle is to learn the true value will knowing the estimated value from an old model, in order to pinpoint where this old model is wrong and to correct it.

## Getting the 4.13 model


```python
from joblib import load
reg = load("gbt_413.joblib")
```


```python
size_columns = ["GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", "vmlinux", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]
```

Getting the columns name ordered for the model.


```python
import json
with open("gbt_413_columns.json","r") as f:
    gbt_413_columns = json.load(f)
```

## Importing the 4.15 version dataset


```python
import pandas as pd
df_415 = pd.read_pickle("datasets/dataset_415.pkl")
```

Assigning a value to the 4.15 dataset in the columns that disappeared to make it compatible with the 4.13 model.


```python
columns_413 = set(gbt_413_columns)
columns_415 = set(df_415.columns.values)

for c in columns_413.difference(columns_415):
    df_415 = df_415.assign(**{c:1})
```

Predicting the value of kernel size for 4.15 data using the 4.13.


```python
X_test_415 = df_415[gbt_413_columns].drop(columns=size_columns+["cid"], errors="ignore")
y_test_415 = df_415["vmlinux"]

y_pred = reg.predict(X_test_415)

dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test_415)/y_test_415).abs()*100})
error_415 = dfErrorsFold["% error"].mean()
print("MAPE for 4.15 : ", error_415)
```

    MAPE for 4.15 :  20.014497458885717


The error is quite high, and no useable as is.

Adding the estimated kernel size to the dataset : 


```python
df_415["estimated_vmlinux"] = y_pred
```

Creating a training set with 5000 examples : 


```python
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

shift_train_size = 5000
shift_X_train, shift_X_test, shift_y_train, shift_y_test = train_test_split(df_415.drop(columns=size_columns+["cid"], errors="ignore"), df_415["vmlinux"], train_size=shift_train_size)
```


```python
gbt = ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=6, min_samples_split=40, loss="huber")
gbt.fit(shift_X_train, shift_y_train)

y_pred = gbt.predict(shift_X_test)

dfErrorsFold = pd.DataFrame({"% error":((y_pred - shift_y_test)/shift_y_test).abs()*100})
print("shifted MAPE : ", dfErrorsFold["% error"].mean())
```

    shifted MAPE :  6.460185613711549


With this technique, and a very reduced training set, it is possible to get a very good accuracy.

It allows a great increase in accuracy compared to the old model, or a new model only trained on the 5000 new examples.

Using larger training set : 


```python
shift_train_size = 10000
shift_X_train, shift_X_test, shift_y_train, shift_y_test = train_test_split(df_415.drop(columns=size_columns+["cid"], errors="ignore"), df_415["vmlinux"], train_size=shift_train_size)

gbt = ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=6, min_samples_split=40, loss="huber")
gbt.fit(shift_X_train, shift_y_train)

y_pred = gbt.predict(shift_X_test)

dfErrorsFold = pd.DataFrame({"% error":((y_pred - shift_y_test)/shift_y_test).abs()*100})
print("shifted MAPE : ", dfErrorsFold["% error"].mean())
```

    shifted MAPE :  5.952643196088162



```python

```
