# Linux Transfer Learning notebook - Model Reuse

In this notebook, we will measure how efficient is a model trained on a previous version of Linux kernel


```python
import pandas as pd
df_413 = pd.read_pickle("datasets/dataset_413.pkl")
```


```python
size_columns = ["GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", "vmlinux", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]
```

## Without Feature Selection

### Training models on 4.13 data

Splitting the dataset into training and testing set. We will use most of the dataset (90%) for training.


```python
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_413.drop(columns=size_columns+["cid"], errors="ignore"), df_413["vmlinux"], train_size=train_size)
```

Training some Gradient Boosting Trees on 4.13 dataset.


```python
gbt = []

for max_depth in [10,15,20]:
    for min_samples_split in [100,120]:
        reg = ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        gbt.append(reg)
```


```python
for reg in gbt:
    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  5.748417903686002
    MAPE :  5.806114524225892
    MAPE :  5.351033352835162
    MAPE :  5.419867125492005
    MAPE :  5.443962456613272
    MAPE :  5.463141003753506


### Measuring accuracy on 4.15 data

Now that we have accurate models on 4.13, let's measure their accuracy on 4.15


```python
df_415 = pd.read_pickle("datasets/dataset_415.pkl")

columns_413 = set(df_413.columns.values)
columns_415 = set(df_415.columns.values)

df_415_reduced = df_415[columns_413.intersection(columns_415)]

for c in columns_413.difference(columns_415):
    df_415_reduced = df_415_reduced.assign(**{c:1})
    
df_415_reduced = df_415_reduced[df_413.columns]
```


```python
X_test_415 = df_415_reduced.drop(columns=size_columns+["cid"], errors="ignore")
y_test_415 = df_415_reduced["vmlinux"]
```


```python
for reg in gbt:
    y_pred = reg.predict(X_test_415)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test_415)/y_test_415).abs()*100})
    print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  19.51873457812496
    MAPE :  19.316773065449517
    MAPE :  18.009820189703913
    MAPE :  19.4311655531924
    MAPE :  20.625966250007096
    MAPE :  20.064375019742883


With around 19% of MAPE, the drop in accuracy is high.

### Training models on 4.15 data

What happens when we create models on a small subset of 4.15 dataset?


```python
train_size = 5000
X_train, X_test, y_train, y_test = train_test_split(df_415.drop(columns=size_columns+["cid"], errors="ignore"), df_415["vmlinux"], train_size=train_size)
```


```python
gbt_415 = []

for max_depth in [10,15,20]:
    for min_samples_split in [100,110]:
        reg = ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        gbt_415.append(reg)
        
        y_pred = reg.predict(X_test)

        dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
        print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  12.451274698791465
    MAPE :  12.699376904645925
    MAPE :  12.344858902143821
    MAPE :  12.703073855965528
    MAPE :  12.600435352748892
    MAPE :  12.756038524831082



```python
rf_415 = []

for max_depth in [10,15,20]:
    for min_samples_split in [5,10]:
        reg = ensemble.RandomForestRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        rf_415.append(reg)
        
        y_pred = reg.predict(X_test)

        dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
        print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  16.051866017399558
    MAPE :  16.0495295292039
    MAPE :  15.633797930375403
    MAPE :  15.731060099086475
    MAPE :  15.516226685933393
    MAPE :  15.596609010372102


With that little data, error rate is still high, but better than models trained on ~85k examples from 4.13

## Using feature selection

### Training models on 4.13 data


```python
import json
with open("feature_ranking_list.json","r") as f:
    feature_ranking_list = json.load(f)
```


```python
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_413[feature_ranking_list[:1500]], df_413["vmlinux"], train_size=train_size)
```


```python
gbt_fs = []

for max_depth in [10,15,20]:
    for min_samples_split in [100,120]:
        reg = ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        gbt_fs.append(reg)
        
        y_pred = reg.predict(X_test)

        dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
        print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  5.947161509251938
    MAPE :  5.91347519058521
    MAPE :  5.553199227362323
    MAPE :  5.555897257663443
    MAPE :  5.577435355690064
    MAPE :  5.5954777235625945


### Measuring accuracy on 4.15 data


```python
X_test_415 = df_415_reduced.drop(columns=size_columns+["cid"], errors="ignore")[feature_ranking_list[:1500]]
y_test_415 = df_415_reduced["vmlinux"]
```


```python
for reg in gbt_fs:
    y_pred = reg.predict(X_test_415)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test_415)/y_test_415).abs()*100})
    print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  16.452639126911276
    MAPE :  17.17782212092812
    MAPE :  16.440303191448237
    MAPE :  15.858597758323798
    MAPE :  16.672099089470418
    MAPE :  16.94841784016877


We can see that the feature selection has a good influence on the accuracy on 4.15, but still not enough.

### Training models on 4.15 data

Using the feature selection directly on a model trained on 4.15 data : 


```python
with open("feature_ranking_list_415.json","r") as f:
    feature_ranking_list_415 = json.load(f)
```


```python
train_size = 5000
X_train, X_test, y_train, y_test = train_test_split(df_415[feature_ranking_list_415[:1500]], df_415["vmlinux"], train_size=train_size)
```


```python
gbt_fs_415 = []

for max_depth in [10,15,20]:
    for min_samples_split in [100,110]:
        reg = ensemble.GradientBoostingRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        gbt_fs_415.append(reg)
        
        y_pred = reg.predict(X_test)

        dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
        print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  10.906124963734033
    MAPE :  10.627949495074306
    MAPE :  10.736555720720435
    MAPE :  10.825859922426863
    MAPE :  10.719017301290679
    MAPE :  10.708744192199589



```python
rf_fs_415 = []

for max_depth in [10,15,20]:
    for min_samples_split in [5,10]:
        reg = ensemble.RandomForestRegressor(n_estimators=50, max_depth=max_depth, min_samples_split=min_samples_split)
        reg.fit(X_train, y_train)
        rf_fs_415.append(reg)
        
        y_pred = reg.predict(X_test)

        dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
        print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  14.921914512983227
    MAPE :  14.836444634379811
    MAPE :  14.086638287846917
    MAPE :  14.289549916313362
    MAPE :  14.096930755592792
    MAPE :  14.168525003760703


In both Gradient Boosting Trees and Random Forest, feature selection makes the models gain in accuracy, and still better than simply using the 4.13 models.

## Conclusion

Models trained on the version 4.13 of Linux Kernel can't be used on other versions as is, the drop in accuracy is too important. Using 5k examples from version 4.15 shows better results than using 85k examples from version 4.13.


```python

```
