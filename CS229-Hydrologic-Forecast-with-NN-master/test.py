import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import format_data as fd

rainoff = pd.read_table('rainfall_collins_la.txt', sep='\s', header=None,index_col=None,engine='python')
runoff = pd.read_table('runoff_collins_la.txt', sep='\s', header=None,index_col=None,engine='python')


X0 = np.array(rainoff.iloc[0:-2])[:,1]
X1 = np.array(rainoff.iloc[1:-1])[:,1]
Y0 = np.array(runoff.iloc[6:-2])[:,1]
Y1 = np.array(runoff.iloc[7:-1])[:,1]
X = np.vstack((X0,X1,Y0,Y1))
print(X)