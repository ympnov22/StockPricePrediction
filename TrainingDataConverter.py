import pandas as pd
import numpy as np
import datetime

pd_data = pd.read_csv( 'StockDataRaw.csv' , index_col = 0, header = 1)

pd_data['Close.6-Open.6'] = pd_data['Close.6'] - pd_data['Open.6']

pd_data_norm = pd_data.apply(lambda x: (x/x.std()), axis=0).fillna(0)

pd_data_diff = pd_data.diff(periods=1)

pd_data_diff_dn = pd_data_diff.dropna()


pd_data_diff_dn_norm = pd_data_diff_dn.apply(lambda x: (x/x.std()), axis=0).fillna(0)

pd_data_concat = pd.concat([pd_data_diff_dn_norm,pd_data_norm['Close.6-Open.6']] , axis=1)

pd_data_concat_dn = pd_data_concat.dropna()

#pd_data_concat_dn.to_csv('TrainingData.csv')

np.savetxt("TrainingData_x.csv", pd_data_concat_dn.values[:-1,:-2], delimiter=",")
np.savetxt("TrainingData_y.csv", pd_data_concat_dn.values[1:,[41,-1]], delimiter=",")

#print(pd_data_concat_dn)
