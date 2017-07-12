# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:10:37 2017

@author: Xiaoxi
"""

%reset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




'assignment is a view NOT copy'

arr = np.arange(10)

arr_slice = arr[5:8]

arr_slice[1] = 9999 #'both arr and arr_slice changed'

arr.ndim
arr.shape


'zip'
xarr = np.array([1,2,3])
yarr = np.array([11,22,33])

for x,y in zip(xarr, yarr):
    print("x =", x , ", y=" , y)

'plotly'
py.tools.set_credentials_file(username='xiaoxig', api_key='yqdHrHsczksG7modM0WW')
trace = go.Scatter(x=xarr, y=yarr, mode='markers')
py.plotly.plot([trace], filename='numpy-arange')

'plt'
plt.plot(xarr, yarr)
plt.ylabel('sts')
plt.show()



'algebra'
arr = np.random.randn(2,4)
np.where(arr > 0, 1, 0)
np.where(arr > 0, 1, arr)
np.mean(arr)
arr.mean()
arr.mean(axis=0)
arr.sum()
arr.cumsum(axis=0)
(arr>0).sum() '# >0'

x,y=np.where(arr>1)


#'data frame'
data = {'state': ['VIC','NSW','WA'],
        'number':[42323,34,14]}

frame = pd.DataFrame(data, index=[1,2,3], columns= ['state','number'])    

frame[1:]    

#'read csv'
path = "C:/Users/Xiaoxi/Dropbox/work/2017/Others/python"
aus_postcode = pd.read_csv(path+'/data/geo_postcode_locations_lat_long.csv',
                           index_col = 0)
                           
type(aus_postcode)
aus_postcode.head(n = 10)

'df - slice'
n = 20
aus_postcode[:n]  
aus_postcode[["postcode", "state"]]
aus_postcode.state[:n]


aus_postcode.loc[1,]  # slice by row and column name
aus_postcode.loc[1, ["postcode", 'state']] 


# 'Select rows by row and column number'
aus_postcode.iloc[:1, :2]  #  Select every row up to n

# 'Select rows FIRST by row and column number, THEN by name'
aus_postcode.ix[2:n, :2] 
aus_postcode.ix[:n, ["postcode", 'state'] ] 


                        
#
aus_postcode_slice = aus_postcode.ix[(aus_postcode.postcode <1000)&(aus_postcode.state == "NT"),
                ["postcode",'state']].copy(deep  = True)   

aus_postcode_slice.shape           
aus_postcode_slice.describe()          


'df - change col type'
aus_postcode.dtypes
aus_postcode.shape
aus_postcode.describe()
aus_postcode.info()
aus_postcode.head(n=10)
aus_postcode.tail(n = 10)

aus_postcode[["lat", "lon"]] = aus_postcode[["lat", "lon"]].apply(pd.to_numeric, errors='coerce')
aus_postcode[["state","suburb"]] = aus_postcode[["state","suburb"]].apply(str) #?
aus_postcode[["postcode"]] = aus_postcode[["postcode"]].astype('str')  # ?


'group by'
groupby_state = aus_postcode.groupby('state')
groupby_state.describe()


groupby_state.agg([np.std, np.mean, len])



'missing value'
test_data = DataFrame([[1,3.1,41],
                      [1,NA,1])