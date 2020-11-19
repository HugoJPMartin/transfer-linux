# Linux Transfer Learning notebook - Boilerplate

## Datasets

Downloading the datasets : 


```python
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_413.pkl -P ./datasets
```

    --2020-07-29 15:10:51--  http://37.187.140.181/transfer_linux_dataset/dataset_413.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 892042316 (851M) [application/octet-stream]
    Saving to: ‘./datasets/dataset_413.pkl’
    
    dataset_413.pkl     100%[===================>] 850.72M  1.13GB/s    in 0.7s    
    
    2020-07-29 15:10:52 (1.13 GB/s) - ‘./datasets/dataset_413.pkl’ saved [892042316/892042316]
    



```python
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_415.pkl -P ./datasets
```

    --2020-07-30 16:08:20--  http://37.187.140.181/transfer_linux_dataset/dataset_415.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 304 Not Modified
    File ‘./datasets/dataset_415.pkl’ not modified on server. Omitting download.
    



```python
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_420.pkl -P ./datasets
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_500.pkl -P ./datasets
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_504.pkl -P ./datasets
!wget -N http://37.187.140.181/transfer_linux_dataset/dataset_507.pkl -P ./datasets
```

    --2020-08-20 07:50:11--  http://37.187.140.181/transfer_linux_dataset/dataset_420.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 304 Not Modified
    File ‘./datasets/dataset_420.pkl’ not modified on server. Omitting download.
    
    --2020-08-20 07:50:11--  http://37.187.140.181/transfer_linux_dataset/dataset_500.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 304 Not Modified
    File ‘./datasets/dataset_500.pkl’ not modified on server. Omitting download.
    
    --2020-08-20 07:50:11--  http://37.187.140.181/transfer_linux_dataset/dataset_504.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 284040300 (271M) [application/octet-stream]
    Saving to: ‘./datasets/dataset_504.pkl’
    
    dataset_504.pkl     100%[===================>] 270.88M  1.00GB/s    in 0.3s    
    
    2020-08-20 07:50:11 (1.00 GB/s) - ‘./datasets/dataset_504.pkl’ saved [284040300/284040300]
    
    --2020-08-20 07:50:12--  http://37.187.140.181/transfer_linux_dataset/dataset_507.pkl
    Connecting to 37.187.140.181:80... connected.
    HTTP request sent, awaiting response... 304 Not Modified
    File ‘./datasets/dataset_507.pkl’ not modified on server. Omitting download.
    


Importing the pickle files into dataframes : 


```python
import pandas as pd
df_413 = pd.read_pickle("datasets/dataset_413.pkl")
df_415 = pd.read_pickle("datasets/dataset_415.pkl")
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
<p>92562 rows × 9487 columns</p>
</div>




```python
df_415
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
      <th>NETFILTER_XT_MATCH_CONNMARK</th>
      <th>NET_EMATCH</th>
      <th>TOUCHSCREEN_AD7877</th>
      <th>REGULATOR_88PM8607</th>
      <th>DVB_USB_CXUSB</th>
      <th>CRYPTO_SHA512_MB</th>
      <th>NETFILTER_XT_MATCH_CONNTRACK</th>
      <th>LAPBETHER</th>
      <th>TOUCHSCREEN_AD7879</th>
      <th>REGULATOR_ACT8865</th>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>7704752</td>
      <td>9791464</td>
      <td>7554872</td>
      <td>12787888</td>
      <td>14874088</td>
      <td>12650651</td>
      <td>13766832</td>
      <td>15853424</td>
      <td>13618427</td>
      <td>29787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
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
      <td>7677200</td>
      <td>9766192</td>
      <td>7453912</td>
      <td>12244240</td>
      <td>14332720</td>
      <td>12029959</td>
      <td>13612304</td>
      <td>15701184</td>
      <td>13389490</td>
      <td>29788</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>13173328</td>
      <td>15259176</td>
      <td>13024548</td>
      <td>21803600</td>
      <td>23888936</td>
      <td>21666603</td>
      <td>23593552</td>
      <td>25679288</td>
      <td>23445627</td>
      <td>29789</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>5633072</td>
      <td>7718672</td>
      <td>5487408</td>
      <td>9552944</td>
      <td>11638024</td>
      <td>9415719</td>
      <td>10482736</td>
      <td>12568216</td>
      <td>10336659</td>
      <td>29790</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
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
      <td>7124016</td>
      <td>9209616</td>
      <td>6979100</td>
      <td>11936816</td>
      <td>14021896</td>
      <td>11802122</td>
      <td>12969008</td>
      <td>15054488</td>
      <td>12822792</td>
      <td>29791</td>
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
      <th>39386</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3134416</td>
      <td>5221256</td>
      <td>2925332</td>
      <td>4867024</td>
      <td>6953352</td>
      <td>4668518</td>
      <td>5411792</td>
      <td>7498512</td>
      <td>5202868</td>
      <td>69996</td>
    </tr>
    <tr>
      <th>39387</th>
      <td>0</td>
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
      <td>5034032</td>
      <td>7120656</td>
      <td>4890576</td>
      <td>8368176</td>
      <td>10454280</td>
      <td>8233143</td>
      <td>9170992</td>
      <td>11257496</td>
      <td>9023391</td>
      <td>69997</td>
    </tr>
    <tr>
      <th>39388</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>15770160</td>
      <td>17857680</td>
      <td>15560416</td>
      <td>28881456</td>
      <td>30968456</td>
      <td>28683704</td>
      <td>32551472</td>
      <td>34638872</td>
      <td>32340626</td>
      <td>69998</td>
    </tr>
    <tr>
      <th>39389</th>
      <td>0</td>
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
      <td>6079952</td>
      <td>8165040</td>
      <td>5931960</td>
      <td>10577360</td>
      <td>12661928</td>
      <td>10443053</td>
      <td>11613648</td>
      <td>13698616</td>
      <td>11465752</td>
      <td>69999</td>
    </tr>
    <tr>
      <th>39390</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4374608</td>
      <td>6461320</td>
      <td>4222564</td>
      <td>7020624</td>
      <td>9106816</td>
      <td>6881019</td>
      <td>7667792</td>
      <td>9754384</td>
      <td>7515896</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
<p>39391 rows × 9444 columns</p>
</div>



## Solving features shifting


```python
columns_413 = set(df_413.columns.values)
columns_415 = set(df_415.columns.values)

print("Number of features in 4.13 dataset : ", len(columns_413))
print("Number of features in 4.15 dataset : ", len(columns_415))
print("Number of features common to both datasets : ", len(columns_413.intersection(columns_415)))
```

    Number of features in 4.13 dataset :  9487
    Number of features in 4.15 dataset :  9444
    Number of features common to both datasets :  9145


We can see that some features appear in 4.15, and other disappear.


```python
print("Number of features that disappear in 4.15 : ", len(columns_413.difference(columns_415)))
print("Number of features that appear in 4.15 : ", len(columns_415.difference(columns_413)))
```

    Number of features that disappear in 4.15 :  342
    Number of features that appear in 4.15 :  299


Note that this is biased by the fact the features with only one value in the dataset have been omitted. You'd have to recreate the dataset without this step to get the exact count.

If we want to align the 4.15 dataset to the 4.13 dataset, we have to perform 2 operations : 
 * Delete "new" features
 * Assign a value to the "old" features

Eliminating the new features is quite easy, only select the features intersection : 


```python
df_415_reduced = df_415[columns_413.intersection(columns_415)]
```

To assign a value for all old features, we need to create the columns one by one and to decide which value to take. Here we will asign the value 1, the other possibility would be 0.


```python
for c in columns_413.difference(columns_415):
    df_415_reduced = df_415_reduced.assign(**{c:1})
```


```python
df_415_reduced
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
      <th>6LOWPAN_GHC_UDP</th>
      <th>LAPB</th>
      <th>PKCS7_TEST_KEY</th>
      <th>SENSORS_W83L786NG</th>
      <th>USB_EHSET_TEST_FIXTURE</th>
      <th>TOUCHSCREEN_TI_AM335X_TSC</th>
      <th>MLXSW_SWITCHIB</th>
      <th>LEDS_PCA963X</th>
      <th>PHY_ROCKCHIP_TYPEC</th>
      <th>IP_SET_LIST_SET</th>
      <th>...</th>
      <th>SCx200_GPIO</th>
      <th>CLONE_BACKWARDS</th>
      <th>SPARSEMEM_MANUAL</th>
      <th>LGUEST_GUEST</th>
      <th>SND_SBAWE</th>
      <th>BLK_DEV_DTC2278</th>
      <th>SCSI_AHA152X</th>
      <th>HIGHMEM64G</th>
      <th>HAVE_VIRT_CPU_ACCOUNTING_GEN</th>
      <th>COPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
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
      <th>39386</th>
      <td>0</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39387</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39388</th>
      <td>0</td>
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
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39389</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39390</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>39391 rows × 9487 columns</p>
</div>




```python
print("Number of features common to 4.13 dataset and the aligned 4.15 dataset : ", len(columns_413.intersection(df_415_reduced)))
```

    Number of features common to 4.13 dataset and the aligned 4.15 dataset :  9487


Reordering the new 4.15 dataset : 


```python
df_415_reduced = df_415_reduced[df_413.columns]
```


```python
df_415_reduced
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
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7704752</td>
      <td>9791464</td>
      <td>7554872</td>
      <td>12787888</td>
      <td>14874088</td>
      <td>12650651</td>
      <td>13766832</td>
      <td>15853424</td>
      <td>13618427</td>
      <td>29787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7677200</td>
      <td>9766192</td>
      <td>7453912</td>
      <td>12244240</td>
      <td>14332720</td>
      <td>12029959</td>
      <td>13612304</td>
      <td>15701184</td>
      <td>13389490</td>
      <td>29788</td>
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
      <td>13173328</td>
      <td>15259176</td>
      <td>13024548</td>
      <td>21803600</td>
      <td>23888936</td>
      <td>21666603</td>
      <td>23593552</td>
      <td>25679288</td>
      <td>23445627</td>
      <td>29789</td>
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
      <td>5633072</td>
      <td>7718672</td>
      <td>5487408</td>
      <td>9552944</td>
      <td>11638024</td>
      <td>9415719</td>
      <td>10482736</td>
      <td>12568216</td>
      <td>10336659</td>
      <td>29790</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7124016</td>
      <td>9209616</td>
      <td>6979100</td>
      <td>11936816</td>
      <td>14021896</td>
      <td>11802122</td>
      <td>12969008</td>
      <td>15054488</td>
      <td>12822792</td>
      <td>29791</td>
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
      <th>39386</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3134416</td>
      <td>5221256</td>
      <td>2925332</td>
      <td>4867024</td>
      <td>6953352</td>
      <td>4668518</td>
      <td>5411792</td>
      <td>7498512</td>
      <td>5202868</td>
      <td>69996</td>
    </tr>
    <tr>
      <th>39387</th>
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
      <td>5034032</td>
      <td>7120656</td>
      <td>4890576</td>
      <td>8368176</td>
      <td>10454280</td>
      <td>8233143</td>
      <td>9170992</td>
      <td>11257496</td>
      <td>9023391</td>
      <td>69997</td>
    </tr>
    <tr>
      <th>39388</th>
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
      <td>15770160</td>
      <td>17857680</td>
      <td>15560416</td>
      <td>28881456</td>
      <td>30968456</td>
      <td>28683704</td>
      <td>32551472</td>
      <td>34638872</td>
      <td>32340626</td>
      <td>69998</td>
    </tr>
    <tr>
      <th>39389</th>
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
      <td>6079952</td>
      <td>8165040</td>
      <td>5931960</td>
      <td>10577360</td>
      <td>12661928</td>
      <td>10443053</td>
      <td>11613648</td>
      <td>13698616</td>
      <td>11465752</td>
      <td>69999</td>
    </tr>
    <tr>
      <th>39390</th>
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
      <td>4374608</td>
      <td>6461320</td>
      <td>4222564</td>
      <td>7020624</td>
      <td>9106816</td>
      <td>6881019</td>
      <td>7667792</td>
      <td>9754384</td>
      <td>7515896</td>
      <td>70000</td>
    </tr>
  </tbody>
</table>
<p>39391 rows × 9487 columns</p>
</div>




```python

```
