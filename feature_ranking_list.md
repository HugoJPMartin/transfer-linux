# Linux Transfer Learning notebook - Feature Ranking List

The goal of this notebook is to produce a reliable list of features ranked by their importance regarding the size of the kernel (vmlinux).

Importing the dataset


```python
import pandas as pd
df_413 = pd.read_pickle("datasets/dataset_413.pkl")
```


```python
df_413
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X86_LOCAL_APIC</th>
      <th>OPENVSWITCH</th>
      <th>TEXTSEARCH_FSM</th>
      <th>NETFILTER_XT_MATCH_TCPMSS</th>
      <th>MPLS</th>
      <th>NFC_HCI</th>
      <th>NETFILTER_XT_MATCH_TIME</th>
      <th>NET_MPLS_GSO</th>
      <th>NFC_SHDLC</th>
      <th>NETFILTER_XT_MATCH_U32</th>
      <th>...</th>
      <th>XZ-bzImage</th>
      <th>XZ-vmlinux</th>
      <th>XZ</th>
      <th>LZO-bzImage</th>
      <th>LZO-vmlinux</th>
      <th>LZO</th>
      <th>LZ4-bzImage</th>
      <th>LZ4-vmlinux</th>
      <th>LZ4</th>
      <th>cid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>5178320.0</td>
      <td>7264848</td>
      <td>4980068</td>
      <td>8922064.0</td>
      <td>11008072</td>
      <td>8734199</td>
      <td>9839568.0</td>
      <td>11925896</td>
      <td>9638560</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2840016.0</td>
      <td>4924448</td>
      <td>2695928</td>
      <td>4519376.0</td>
      <td>6603288</td>
      <td>4385061</td>
      <td>4838864.0</td>
      <td>6923096</td>
      <td>4693085</td>
      <td>30001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8496592.0</td>
      <td>10581024</td>
      <td>8351248</td>
      <td>12391888.0</td>
      <td>14475800</td>
      <td>12256864</td>
      <td>13362640.0</td>
      <td>15446872</td>
      <td>13214970</td>
      <td>30002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6304720.0</td>
      <td>8390008</td>
      <td>6156724</td>
      <td>8782800.0</td>
      <td>10867576</td>
      <td>8647251</td>
      <td>9302992.0</td>
      <td>11388080</td>
      <td>9155423</td>
      <td>30003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>12321744.0</td>
      <td>14407032</td>
      <td>12176312</td>
      <td>17933264.0</td>
      <td>20018040</td>
      <td>17796721</td>
      <td>19346384.0</td>
      <td>21431472</td>
      <td>19197696</td>
      <td>30004</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>92557</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552400.0</td>
      <td>2638880</td>
      <td>411384</td>
      <td>691664.0</td>
      <td>2777624</td>
      <td>558713</td>
      <td>724432.0</td>
      <td>2810712</td>
      <td>578376</td>
      <td>126756</td>
    </tr>
    <tr>
      <th>92558</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552400.0</td>
      <td>2638880</td>
      <td>411312</td>
      <td>691664.0</td>
      <td>2777624</td>
      <td>558713</td>
      <td>724432.0</td>
      <td>2810712</td>
      <td>578376</td>
      <td>126757</td>
    </tr>
    <tr>
      <th>92559</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552400.0</td>
      <td>2638880</td>
      <td>411328</td>
      <td>691664.0</td>
      <td>2777624</td>
      <td>558713</td>
      <td>724432.0</td>
      <td>2810712</td>
      <td>578376</td>
      <td>126758</td>
    </tr>
    <tr>
      <th>92560</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552400.0</td>
      <td>2638880</td>
      <td>411336</td>
      <td>691664.0</td>
      <td>2777624</td>
      <td>558713</td>
      <td>724432.0</td>
      <td>2810712</td>
      <td>578376</td>
      <td>126759</td>
    </tr>
    <tr>
      <th>92561</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>552400.0</td>
      <td>2638880</td>
      <td>411368</td>
      <td>691664.0</td>
      <td>2777624</td>
      <td>558713</td>
      <td>724432.0</td>
      <td>2810712</td>
      <td>578376</td>
      <td>126760</td>
    </tr>
  </tbody>
</table>
<p>92562 rows × 9488 columns</p>
</div>




```python
size_columns = ["GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", "vmlinux", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]
```

Splitting the dataset into training and testing set. We will use most of the dataset (90%) for training.


```python
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_413.drop(columns=size_columns+["cid"], errors="ignore"), df_413["vmlinux"], train_size=train_size)
```

Training a single Random Forest over the training set : 


```python
# Setting hyperparameters for the Random Forest
reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

# Fitting the model
reg.fit(X_train, y_train)

# Predicting the testing set and computing the Mean Average Percentage Error (MAPE)
y_pred = reg.predict(X_test)
dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})

print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  8.332935361473846


A MAPE of 8% is good enough to get a list.


```python
feature_importance = pd.Series(reg.feature_importances_, X_train.columns)
feature_importance.sort_values()
```




    X86_LOCAL_APIC          0.000000
    PNFS_FLEXFILE_LAYOUT    0.000000
    IPW2200_PROMISCUOUS     0.000000
    USB_F_TCM               0.000000
    NFT_CHAIN_NAT_IPV6      0.000000
                              ...   
    X86_NEED_RELOCS         0.082012
    DEBUG_INFO_SPLIT        0.083730
    DEBUG_INFO_REDUCED      0.112955
    active_options          0.189966
    DEBUG_INFO              0.333074
    Length: 9468, dtype: float64




```python
list(feature_importance.sort_values(ascending=False).index)[:20]
```




    ['DEBUG_INFO',
     'active_options',
     'DEBUG_INFO_REDUCED',
     'DEBUG_INFO_SPLIT',
     'X86_NEED_RELOCS',
     'RANDOMIZE_BASE',
     'UBSAN_SANITIZE_ALL',
     'KASAN',
     'KASAN_OUTLINE',
     'UBSAN_ALIGNMENT',
     'GCOV_PROFILE_ALL',
     'DRM_NOUVEAU',
     'XFS_DEBUG',
     'XFS_FS',
     'DRM_RADEON',
     'KCOV_INSTRUMENT_ALL',
     'DRM_AMDGPU',
     'BLK_MQ_PCI',
     'MAXSMP',
     'UBSAN_NULL']



Create another Random Forest with the same parameters to compare the feature list.


```python
# Setting hyperparameters for the Random Forest
reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

# Fitting the model
reg.fit(X_train, y_train)

# Predicting the testing set and computing the Mean Average Percentage Error (MAPE)
y_pred = reg.predict(X_test)
dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})

print("MAPE : ", dfErrorsFold["% error"].mean())
```

    MAPE :  8.329455250921



```python
feature_importance_2 = pd.Series(reg.feature_importances_, X_train.columns)
feature_importance_2.sort_values()
```




    X86_LOCAL_APIC            0.000000
    RT2500PCI                 0.000000
    DWMAC_DWC_QOS_ETH         0.000000
    ATH10K_SDIO               0.000000
    ROADRUNNER_LARGE_RINGS    0.000000
                                ...   
    DEBUG_INFO_SPLIT          0.084633
    RANDOMIZE_BASE            0.088638
    DEBUG_INFO_REDUCED        0.112856
    active_options            0.190650
    DEBUG_INFO                0.334011
    Length: 9468, dtype: float64



If we compare the top 300 features, not even half of them are in both lists.


```python
len(set(list(feature_importance.sort_values(ascending=False).index)[:300]).intersection(set(list(feature_importance_2.sort_values(ascending=False).index)[:300])))
```




    130



Despite being trained on the exact same dataset and with the same hyperparameters, we can't reach a consistent list.

What is commonly done to fight the impact of randomness on experiment is to repeat the same operation multiple times and take the average.

Running 20 Random Forests : 


```python
df_importance = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance = df_importance.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
```

    MAPE 8.285511414502052
    MAPE 8.274145726672739
    MAPE 8.242216798686933
    MAPE 8.244932803008837
    MAPE 8.242479623075914
    MAPE 8.29850871228274
    MAPE 8.309497755535356
    MAPE 8.252582547160882
    MAPE 8.279412430914705
    MAPE 8.27467161620697
    MAPE 8.296036435366942
    MAPE 8.287716557760033
    MAPE 8.312320296409563
    MAPE 8.253339028699319
    MAPE 8.25276033316606
    MAPE 8.306490357823227
    MAPE 8.275159235832254
    MAPE 8.272655252871742
    MAPE 8.293269522151649
    MAPE 8.29017433648496



```python
df_importance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X86_LOCAL_APIC</th>
      <th>OPENVSWITCH</th>
      <th>TEXTSEARCH_FSM</th>
      <th>NETFILTER_XT_MATCH_TCPMSS</th>
      <th>MPLS</th>
      <th>NFC_HCI</th>
      <th>NETFILTER_XT_MATCH_TIME</th>
      <th>NET_MPLS_GSO</th>
      <th>NFC_SHDLC</th>
      <th>NETFILTER_XT_MATCH_U32</th>
      <th>...</th>
      <th>APDS9960</th>
      <th>ARCH_SUPPORTS_INT128</th>
      <th>SLABINFO</th>
      <th>MICROCODE_AMD</th>
      <th>ISDN_DRV_HISAX</th>
      <th>CHARGER_BQ24190</th>
      <th>SND_SOC_NAU8825</th>
      <th>BH1750</th>
      <th>NETWORK_FILESYSTEMS</th>
      <th>active_options</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.947770e-08</td>
      <td>2.711327e-06</td>
      <td>1.421777e-07</td>
      <td>1.819928e-06</td>
      <td>0.000086</td>
      <td>1.882985e-05</td>
      <td>8.295066e-07</td>
      <td>0.000020</td>
      <td>1.515036e-07</td>
      <td>1.904537e-06</td>
      <td>...</td>
      <td>0.000007</td>
      <td>1.435320e-08</td>
      <td>0.000004</td>
      <td>1.004575e-06</td>
      <td>7.572362e-08</td>
      <td>0.000008</td>
      <td>0.000002</td>
      <td>6.802712e-06</td>
      <td>0.000022</td>
      <td>0.190339</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.153610e-08</td>
      <td>1.032557e-05</td>
      <td>3.966991e-07</td>
      <td>2.237942e-05</td>
      <td>0.000048</td>
      <td>1.055633e-05</td>
      <td>0.000000e+00</td>
      <td>0.000005</td>
      <td>5.804795e-07</td>
      <td>2.178889e-08</td>
      <td>...</td>
      <td>0.000040</td>
      <td>0.000000e+00</td>
      <td>0.000013</td>
      <td>3.046460e-06</td>
      <td>2.990701e-07</td>
      <td>0.000017</td>
      <td>0.000001</td>
      <td>3.082698e-06</td>
      <td>0.000014</td>
      <td>0.189601</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000e+00</td>
      <td>3.704904e-05</td>
      <td>3.995143e-06</td>
      <td>1.099438e-09</td>
      <td>0.000114</td>
      <td>1.253737e-05</td>
      <td>0.000000e+00</td>
      <td>0.000007</td>
      <td>1.048344e-05</td>
      <td>4.418103e-08</td>
      <td>...</td>
      <td>0.000068</td>
      <td>1.139941e-08</td>
      <td>0.000004</td>
      <td>3.743083e-06</td>
      <td>2.548254e-08</td>
      <td>0.000013</td>
      <td>0.000002</td>
      <td>7.553984e-07</td>
      <td>0.000009</td>
      <td>0.188627</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.778242e-08</td>
      <td>6.919974e-06</td>
      <td>1.065871e-07</td>
      <td>5.452853e-08</td>
      <td>0.000051</td>
      <td>5.265264e-06</td>
      <td>0.000000e+00</td>
      <td>0.000007</td>
      <td>2.684252e-06</td>
      <td>2.216115e-08</td>
      <td>...</td>
      <td>0.000025</td>
      <td>0.000000e+00</td>
      <td>0.000003</td>
      <td>5.998434e-06</td>
      <td>3.079440e-08</td>
      <td>0.000010</td>
      <td>0.000002</td>
      <td>3.046843e-06</td>
      <td>0.000022</td>
      <td>0.189715</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.306345e-08</td>
      <td>7.985219e-05</td>
      <td>1.335731e-07</td>
      <td>0.000000e+00</td>
      <td>0.000132</td>
      <td>5.909183e-07</td>
      <td>0.000000e+00</td>
      <td>0.000005</td>
      <td>3.887822e-06</td>
      <td>8.636150e-09</td>
      <td>...</td>
      <td>0.000031</td>
      <td>0.000000e+00</td>
      <td>0.000005</td>
      <td>7.140426e-06</td>
      <td>2.454632e-08</td>
      <td>0.000017</td>
      <td>0.000002</td>
      <td>2.469203e-06</td>
      <td>0.000034</td>
      <td>0.188967</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000000e+00</td>
      <td>4.307789e-05</td>
      <td>7.111024e-08</td>
      <td>9.971875e-09</td>
      <td>0.000075</td>
      <td>7.542196e-06</td>
      <td>0.000000e+00</td>
      <td>0.000015</td>
      <td>7.721866e-07</td>
      <td>3.961225e-09</td>
      <td>...</td>
      <td>0.000039</td>
      <td>1.061883e-08</td>
      <td>0.000017</td>
      <td>2.969906e-05</td>
      <td>1.136279e-07</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>3.999839e-06</td>
      <td>0.000032</td>
      <td>0.188504</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000000e+00</td>
      <td>4.773652e-05</td>
      <td>1.305358e-07</td>
      <td>2.772411e-08</td>
      <td>0.000021</td>
      <td>4.478896e-06</td>
      <td>2.495846e-08</td>
      <td>0.000002</td>
      <td>2.038826e-06</td>
      <td>2.916973e-08</td>
      <td>...</td>
      <td>0.000024</td>
      <td>1.084466e-08</td>
      <td>0.000003</td>
      <td>3.755865e-06</td>
      <td>8.237768e-09</td>
      <td>0.000029</td>
      <td>0.000003</td>
      <td>2.074400e-06</td>
      <td>0.000022</td>
      <td>0.189207</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.584593e-08</td>
      <td>1.688000e-05</td>
      <td>7.127066e-08</td>
      <td>8.882972e-09</td>
      <td>0.000110</td>
      <td>6.217624e-06</td>
      <td>2.131213e-08</td>
      <td>0.000009</td>
      <td>2.461935e-06</td>
      <td>3.321757e-06</td>
      <td>...</td>
      <td>0.000019</td>
      <td>7.974841e-09</td>
      <td>0.000014</td>
      <td>4.614817e-06</td>
      <td>7.286357e-09</td>
      <td>0.000021</td>
      <td>0.000002</td>
      <td>4.135085e-06</td>
      <td>0.000005</td>
      <td>0.189837</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000000e+00</td>
      <td>2.351858e-05</td>
      <td>8.218472e-08</td>
      <td>5.528946e-07</td>
      <td>0.000047</td>
      <td>9.299145e-06</td>
      <td>0.000000e+00</td>
      <td>0.000004</td>
      <td>2.531190e-07</td>
      <td>8.179744e-08</td>
      <td>...</td>
      <td>0.000028</td>
      <td>2.634271e-08</td>
      <td>0.000007</td>
      <td>1.510898e-06</td>
      <td>1.130277e-08</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>4.302821e-06</td>
      <td>0.000041</td>
      <td>0.187515</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000000e+00</td>
      <td>5.638109e-05</td>
      <td>1.273168e-07</td>
      <td>5.862267e-08</td>
      <td>0.000064</td>
      <td>8.311707e-06</td>
      <td>1.617544e-08</td>
      <td>0.000012</td>
      <td>2.619293e-06</td>
      <td>4.555317e-08</td>
      <td>...</td>
      <td>0.000025</td>
      <td>9.290595e-09</td>
      <td>0.000003</td>
      <td>5.527940e-06</td>
      <td>3.843503e-09</td>
      <td>0.000011</td>
      <td>0.000002</td>
      <td>2.442374e-06</td>
      <td>0.000026</td>
      <td>0.189877</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000e+00</td>
      <td>2.477614e-05</td>
      <td>5.851329e-06</td>
      <td>0.000000e+00</td>
      <td>0.000018</td>
      <td>1.387216e-06</td>
      <td>8.763644e-06</td>
      <td>0.000030</td>
      <td>3.491765e-06</td>
      <td>8.310331e-08</td>
      <td>...</td>
      <td>0.000038</td>
      <td>0.000000e+00</td>
      <td>0.000007</td>
      <td>1.470565e-06</td>
      <td>6.877981e-08</td>
      <td>0.000012</td>
      <td>0.000001</td>
      <td>5.744651e-06</td>
      <td>0.000048</td>
      <td>0.188616</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000000e+00</td>
      <td>1.249472e-05</td>
      <td>1.190714e-07</td>
      <td>1.075013e-08</td>
      <td>0.000027</td>
      <td>3.470234e-06</td>
      <td>1.846701e-06</td>
      <td>0.000028</td>
      <td>3.107548e-06</td>
      <td>1.537507e-08</td>
      <td>...</td>
      <td>0.000053</td>
      <td>0.000000e+00</td>
      <td>0.000006</td>
      <td>8.760015e-06</td>
      <td>1.958484e-07</td>
      <td>0.000017</td>
      <td>0.000002</td>
      <td>1.866088e-06</td>
      <td>0.000033</td>
      <td>0.188507</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1.270146e-08</td>
      <td>7.376788e-07</td>
      <td>4.558574e-08</td>
      <td>2.622127e-08</td>
      <td>0.000060</td>
      <td>1.570720e-05</td>
      <td>0.000000e+00</td>
      <td>0.000030</td>
      <td>4.280404e-06</td>
      <td>1.568693e-06</td>
      <td>...</td>
      <td>0.000025</td>
      <td>0.000000e+00</td>
      <td>0.000003</td>
      <td>9.997358e-07</td>
      <td>1.134772e-07</td>
      <td>0.000021</td>
      <td>0.000003</td>
      <td>4.043958e-06</td>
      <td>0.000011</td>
      <td>0.190240</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.154798e-08</td>
      <td>3.740128e-05</td>
      <td>2.449126e-05</td>
      <td>8.192016e-08</td>
      <td>0.000055</td>
      <td>2.304631e-05</td>
      <td>2.156860e-08</td>
      <td>0.000014</td>
      <td>2.075654e-06</td>
      <td>1.447895e-06</td>
      <td>...</td>
      <td>0.000040</td>
      <td>2.059247e-09</td>
      <td>0.000007</td>
      <td>3.598679e-06</td>
      <td>1.061297e-07</td>
      <td>0.000016</td>
      <td>0.000002</td>
      <td>5.735317e-06</td>
      <td>0.000017</td>
      <td>0.188651</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.087324e-08</td>
      <td>5.206349e-05</td>
      <td>2.675716e-07</td>
      <td>3.165125e-08</td>
      <td>0.000063</td>
      <td>2.676837e-07</td>
      <td>6.587085e-09</td>
      <td>0.000011</td>
      <td>2.377154e-06</td>
      <td>2.126979e-07</td>
      <td>...</td>
      <td>0.000029</td>
      <td>3.154067e-09</td>
      <td>0.000004</td>
      <td>1.615777e-06</td>
      <td>4.531853e-08</td>
      <td>0.000011</td>
      <td>0.000003</td>
      <td>3.047397e-06</td>
      <td>0.000018</td>
      <td>0.187901</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.083096e-08</td>
      <td>4.441525e-05</td>
      <td>1.425933e-07</td>
      <td>7.948319e-09</td>
      <td>0.000037</td>
      <td>1.825207e-05</td>
      <td>5.029669e-08</td>
      <td>0.000019</td>
      <td>4.518391e-06</td>
      <td>2.373627e-08</td>
      <td>...</td>
      <td>0.000017</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
      <td>3.297323e-06</td>
      <td>1.047574e-07</td>
      <td>0.000009</td>
      <td>0.000001</td>
      <td>3.540166e-06</td>
      <td>0.000029</td>
      <td>0.190967</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.000000e+00</td>
      <td>1.668012e-05</td>
      <td>7.659056e-08</td>
      <td>4.076163e-09</td>
      <td>0.000074</td>
      <td>4.370178e-07</td>
      <td>2.569174e-08</td>
      <td>0.000014</td>
      <td>3.682970e-08</td>
      <td>1.856380e-07</td>
      <td>...</td>
      <td>0.000027</td>
      <td>9.875164e-09</td>
      <td>0.000005</td>
      <td>5.121149e-06</td>
      <td>5.714816e-08</td>
      <td>0.000009</td>
      <td>0.000002</td>
      <td>8.456072e-06</td>
      <td>0.000008</td>
      <td>0.189359</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2.392856e-08</td>
      <td>1.827089e-06</td>
      <td>2.198446e-07</td>
      <td>1.862507e-08</td>
      <td>0.000016</td>
      <td>1.047790e-05</td>
      <td>9.913299e-10</td>
      <td>0.000011</td>
      <td>3.496194e-06</td>
      <td>6.595745e-08</td>
      <td>...</td>
      <td>0.000018</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
      <td>2.555244e-06</td>
      <td>4.828712e-08</td>
      <td>0.000007</td>
      <td>0.000001</td>
      <td>2.804122e-06</td>
      <td>0.000028</td>
      <td>0.188732</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.000000e+00</td>
      <td>2.742456e-05</td>
      <td>7.158084e-07</td>
      <td>8.640361e-09</td>
      <td>0.000054</td>
      <td>5.201705e-06</td>
      <td>0.000000e+00</td>
      <td>0.000031</td>
      <td>1.281949e-06</td>
      <td>1.546127e-06</td>
      <td>...</td>
      <td>0.000032</td>
      <td>8.891912e-09</td>
      <td>0.000013</td>
      <td>1.356242e-06</td>
      <td>1.271818e-08</td>
      <td>0.000014</td>
      <td>0.000002</td>
      <td>8.039783e-06</td>
      <td>0.000007</td>
      <td>0.189820</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.000000e+00</td>
      <td>4.253990e-05</td>
      <td>7.132281e-07</td>
      <td>1.764588e-08</td>
      <td>0.000044</td>
      <td>1.953604e-05</td>
      <td>2.752424e-08</td>
      <td>0.000004</td>
      <td>1.426477e-06</td>
      <td>1.612714e-08</td>
      <td>...</td>
      <td>0.000022</td>
      <td>1.055419e-08</td>
      <td>0.000002</td>
      <td>5.924138e-06</td>
      <td>5.949257e-09</td>
      <td>0.000009</td>
      <td>0.000002</td>
      <td>3.801353e-06</td>
      <td>0.000011</td>
      <td>0.189461</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 9468 columns</p>
</div>



Adding a row being the average of all Random Forests feature importance.


```python
df_importance.loc["mean"] = df_importance.mean()
```

Creating a ranking value instead of arbitrary feature importance value : 


```python
df_values = df_importance.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>ranking-11</th>
      <th>ranking-12</th>
      <th>ranking-13</th>
      <th>ranking-14</th>
      <th>ranking-15</th>
      <th>ranking-16</th>
      <th>ranking-17</th>
      <th>ranking-18</th>
      <th>ranking-19</th>
      <th>ranking-mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DEBUG_INFO</th>
      <td>0.332192</td>
      <td>0.333460</td>
      <td>0.333127</td>
      <td>0.334529</td>
      <td>0.334833</td>
      <td>0.334251</td>
      <td>0.333904</td>
      <td>0.333460</td>
      <td>0.332828</td>
      <td>0.333958</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>active_options</th>
      <td>0.190339</td>
      <td>0.189601</td>
      <td>0.188627</td>
      <td>0.189715</td>
      <td>0.188967</td>
      <td>0.188504</td>
      <td>0.189207</td>
      <td>0.189837</td>
      <td>0.187515</td>
      <td>0.189877</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>DEBUG_INFO_REDUCED</th>
      <td>0.114519</td>
      <td>0.113372</td>
      <td>0.114125</td>
      <td>0.112747</td>
      <td>0.112122</td>
      <td>0.112506</td>
      <td>0.112742</td>
      <td>0.113206</td>
      <td>0.114613</td>
      <td>0.112829</td>
      <td>...</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>DEBUG_INFO_SPLIT</th>
      <td>0.085347</td>
      <td>0.085117</td>
      <td>0.085581</td>
      <td>0.084328</td>
      <td>0.084896</td>
      <td>0.085363</td>
      <td>0.084545</td>
      <td>0.085443</td>
      <td>0.087078</td>
      <td>0.084405</td>
      <td>...</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>RANDOMIZE_BASE</th>
      <td>0.065824</td>
      <td>0.077844</td>
      <td>0.081720</td>
      <td>0.079105</td>
      <td>0.069965</td>
      <td>0.077121</td>
      <td>0.078492</td>
      <td>0.070109</td>
      <td>0.088558</td>
      <td>0.073020</td>
      <td>...</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>SND_SBAWE_SEQ</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>8953.0</td>
      <td>8958.0</td>
      <td>8952.0</td>
      <td>8969.0</td>
      <td>8949.0</td>
      <td>8950.0</td>
      <td>8940.0</td>
      <td>8946.0</td>
      <td>8966.0</td>
      <td>9334.0</td>
    </tr>
    <tr>
      <th>MIXCOMWD</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>8953.0</td>
      <td>8958.0</td>
      <td>8952.0</td>
      <td>8969.0</td>
      <td>8949.0</td>
      <td>8950.0</td>
      <td>8940.0</td>
      <td>8946.0</td>
      <td>8966.0</td>
      <td>9334.0</td>
    </tr>
    <tr>
      <th>PCWATCHDOG</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>8953.0</td>
      <td>8958.0</td>
      <td>8952.0</td>
      <td>8969.0</td>
      <td>8949.0</td>
      <td>8950.0</td>
      <td>8940.0</td>
      <td>8946.0</td>
      <td>8966.0</td>
      <td>9334.0</td>
    </tr>
    <tr>
      <th>SND_SB8</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>8953.0</td>
      <td>8958.0</td>
      <td>8952.0</td>
      <td>8969.0</td>
      <td>8949.0</td>
      <td>8950.0</td>
      <td>8940.0</td>
      <td>8946.0</td>
      <td>8966.0</td>
      <td>9334.0</td>
    </tr>
    <tr>
      <th>LGUEST_GUEST</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>8953.0</td>
      <td>8958.0</td>
      <td>8952.0</td>
      <td>8969.0</td>
      <td>8949.0</td>
      <td>8950.0</td>
      <td>8940.0</td>
      <td>8946.0</td>
      <td>8966.0</td>
      <td>9334.0</td>
    </tr>
  </tbody>
</table>
<p>9468 rows × 42 columns</p>
</div>



Here is the feature ranking list and displaying the top 20 : 


```python
feature_ranking_list = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)
feature_ranking_list[:20]
```




    ['DEBUG_INFO',
     'active_options',
     'DEBUG_INFO_REDUCED',
     'DEBUG_INFO_SPLIT',
     'RANDOMIZE_BASE',
     'X86_NEED_RELOCS',
     'UBSAN_SANITIZE_ALL',
     'KASAN',
     'KASAN_OUTLINE',
     'UBSAN_ALIGNMENT',
     'GCOV_PROFILE_ALL',
     'DRM_NOUVEAU',
     'XFS_DEBUG',
     'XFS_FS',
     'KCOV_INSTRUMENT_ALL',
     'UBSAN_NULL',
     'MAXSMP',
     'DRM_RADEON',
     'DRM_AMDGPU',
     'BLK_MQ_PCI']



Repeating the operationwith 20 new Random Forests : 


```python
df_importance_2 = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance_2 = df_importance_2.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
```

    MAPE 8.287202180072523
    MAPE 8.268799452000907
    MAPE 8.269959063607905
    MAPE 8.230731081761608
    MAPE 8.25857992098621
    MAPE 8.272208631122407
    MAPE 8.256220039656345
    MAPE 8.27052677409201
    MAPE 8.309480808728713
    MAPE 8.258349634800531
    MAPE 8.321599424168372
    MAPE 8.266521859342685
    MAPE 8.242671467628158
    MAPE 8.263093950175165
    MAPE 8.289319022203797
    MAPE 8.244905469823037
    MAPE 8.27170823500447
    MAPE 8.297003691270122
    MAPE 8.291465965584294
    MAPE 8.284461482230416



```python
df_importance_2.loc["mean"] = df_importance_2.mean()

df_values = df_importance_2.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")

feature_ranking_list_2 = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)

```

On the top 300, we get a much more consistent list : 


```python
len(set(feature_ranking_list[:300]).intersection(set(feature_ranking_list_2[:300])))
```




    249



Exporting the Feature Ranking List : 


```python
import json
with open("feature_ranking_list.json","w") as f:
    json.dump(feature_ranking_list, f)
```

## Repeating the process for 4.15


```python
df_415 = pd.read_pickle("datasets/dataset_415.pkl")
```


```python
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_415.drop(columns=size_columns+["cid"], errors="ignore"), df_415["vmlinux"], train_size=train_size)
```


```python
df_importance_415 = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance_415 = df_importance_415.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
```

    MAPE 9.679986650260258
    MAPE 9.745662923246531
    MAPE 9.64720082211712
    MAPE 9.676742159201646
    MAPE 9.69135123212296
    MAPE 9.712152074772114
    MAPE 9.763815968122232
    MAPE 9.738455313284804
    MAPE 9.736208395714042
    MAPE 9.659228226023227
    MAPE 9.705100769878776
    MAPE 9.749813265843999
    MAPE 9.71220417157918
    MAPE 9.732416282881617
    MAPE 9.699609358357515
    MAPE 9.711474810673192
    MAPE 9.682008213397783
    MAPE 9.716561093573524
    MAPE 9.675108662698907
    MAPE 9.716774405109081



```python
df_importance_415.loc["mean"] = df_importance_415.mean()

df_values = df_importance_415.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")

feature_ranking_list_415 = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)

```


```python
with open("feature_ranking_list_415.json","w") as f:
    json.dump(feature_ranking_list_415, f)
```

## 4.20


```python
import pandas as pd
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split

import json
```


```python
df_420 = pd.read_pickle("datasets/dataset_420.pkl")
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_420.drop(columns=size_columns+["cid"], errors="ignore"), df_420["vmlinux"], train_size=train_size)

df_importance_420 = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance_420 = df_importance_420.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
    
    
df_importance_420.loc["mean"] = df_importance_420.mean()

df_values = df_importance_420.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")

feature_ranking_list_420 = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)

with open("feature_ranking_list_420.json","w") as f:
    json.dump(feature_ranking_list_420, f)
```

    MAPE 10.408788395968958
    MAPE 10.375449127208253
    MAPE 10.466458064581339
    MAPE 10.474524729893206
    MAPE 10.409071564691823
    MAPE 10.301506176773048
    MAPE 10.260633449664935
    MAPE 10.326733579382935
    MAPE 10.229475952867851
    MAPE 10.414943267165265
    MAPE 10.33619419726815
    MAPE 10.48977394578446
    MAPE 10.33983046047351
    MAPE 10.42414138725688
    MAPE 10.377826217523369
    MAPE 10.294373383590266
    MAPE 10.383342049861167
    MAPE 10.354220917147726
    MAPE 10.346926214742618
    MAPE 10.35969477323931


## 5.0


```python
df_500 = pd.read_pickle("datasets/dataset_420.pkl")
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_500.drop(columns=size_columns+["cid"], errors="ignore"), df_500["vmlinux"], train_size=train_size)

df_importance_500 = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance_500 = df_importance_500.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
    
    
df_importance_500.loc["mean"] = df_importance_420.mean()

df_values = df_importance_500.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")

feature_ranking_list_500 = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)

with open("feature_ranking_list_500.json","w") as f:
    json.dump(feature_ranking_list_500, f)
```

    MAPE 10.726184430037451
    MAPE 10.847330679877034
    MAPE 10.719863684174282
    MAPE 10.872867766133757
    MAPE 10.635848866576575
    MAPE 10.575122364308871
    MAPE 10.793602469913706
    MAPE 10.810060719135617
    MAPE 10.703972884550582
    MAPE 10.715310334399907
    MAPE 10.736297614153976
    MAPE 10.818871795458369
    MAPE 10.796061408097898
    MAPE 10.767850819675402
    MAPE 10.594468121400084
    MAPE 10.736107990162827
    MAPE 10.741051368304563
    MAPE 10.76241855327311
    MAPE 10.7486353601307
    MAPE 10.737202492394998


## 5.7


```python
df_507 = pd.read_pickle("datasets/dataset_420.pkl")
train_size = 0.9
X_train, X_test, y_train, y_test = train_test_split(df_507.drop(columns=size_columns+["cid"], errors="ignore"), df_507["vmlinux"], train_size=train_size)

df_importance_507 = pd.DataFrame()

for _ in range(0,20):
    reg = ensemble.RandomForestRegressor(n_estimators=48, max_depth=20, min_samples_split=10, n_jobs=8)

    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    dfErrorsFold = pd.DataFrame({"% error":((y_pred - y_test)/y_test).abs()*100})
    print("MAPE", dfErrorsFold["% error"].mean())

    df_importance_507 = df_importance_507.append(pd.DataFrame([reg.feature_importances_], columns=X_train.columns), ignore_index=True)
    
    
df_importance_507.loc["mean"] = df_importance_420.mean()

df_values = df_importance_507.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
df_values.sort_values("ranking-mean")

feature_ranking_list_507 = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)

with open("feature_ranking_list_507.json","w") as f:
    json.dump(feature_ranking_list_507, f)
```

    MAPE 10.19838897143361
    MAPE 10.433352870629836
    MAPE 10.215187764538127
    MAPE 10.259523955771101
    MAPE 10.281615299336915
    MAPE 10.336696387940219
    MAPE 10.305257480253381
    MAPE 10.242511125922487
    MAPE 10.296641309437346
    MAPE 10.179605423559591
    MAPE 10.303374955474252
    MAPE 10.230022039238689
    MAPE 10.323502837156923
    MAPE 10.302966203824313
    MAPE 10.135533029425279
    MAPE 10.347721026049884
    MAPE 10.343018412897518
    MAPE 10.284218555626234
    MAPE 10.220949351856516
    MAPE 10.38249549898559



```python

```
