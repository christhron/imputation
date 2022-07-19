#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from scipy.signal import savgol_filter

# Options for 2007 data
dfName = 'Nal_filtre_FLX_NEE_01012007_31122010.csv'
firstEntry = 10000

# Options for 2011 data
dfName = 'Nal_filtre_FLX_NEE_01012011_31122017.csv'
firstEntry = 0

df = pd.read_csv(os.path.abspath(dfName), encoding = "ISO-8859-1", sep = ';')

# Set x limits for plot
xi = np.arange(1000,2000)



# In[17]:


df.head(5)


# In[18]:


print(df.columns)


# fixing the name to be the 1st row

# In[19]:


df.columns = df.iloc[0] # fix the col names to be the 1st col
df = df.iloc[1:] # fix the col names to be the 1st col


# In[20]:


df.head(2)


# In[21]:


df.shape


# Checking the type

# In[22]:


df.dtypes


# we need to change the following [2:] to float64

# In[14]:


df.iloc[:, 2:] = df.iloc[:, 2:].astype(float)

df


# In[30]:


# Note that first 10000 rows are all NaNs
df.iloc[10000,:]


# In[ ]:


### Data manipulations:

# * Remove first 10000 rows
# * Convert to numpy array
# * Create logical array for NaN's
# * Remove outliers from last 2 columns


# In[58]:


X = df.iloc[firstEntry:, 2:].to_numpy()
X = X.astype(float)
Xout = np.abs(X[:,-2:])>50
X[Xout[:,0],-2] = float("nan")
X[Xout[:,1],-1] = float("nan")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   
X = scaler.fit_transform(X)

# Find nan's
Xnan = np.isnan(X)
# Replace nan's with 0
X0 = np.copy(X)
X0[Xnan] = 0
# Integer matrix that indicates data locations
Xan = 1 - Xnan.astype(int)

# Matrix shape
(nx, nc) = np.shape(Xnan)


# ## Graphs
# 
# * Graphs show evident daily  and yearly periodicity (4 years of data)

# In[64]:

# Set xi if not set
if len(xi)==0:
    xi = np.arange(nx)

for jj,col in enumerate(df.columns[2:]):
    plt.plot(xi,X[xi,jj])
    nanLoc = np.array(np.where(Xnan[xi,jj]))
    print(np.shape(nanLoc))
    plt.plot(xi[nanLoc],0*nanLoc,'r.')   
    plt.title(col)
    plt.show()

fig, axs = plt.subplots(nc-1, nc-1)
for jj in range(nc):
    for kk in range(jj+1,nc):
        axs[jj, kk-1].scatter(X[:,jj], X[:,kk],c='k',s=0.25)
        axs[jj, kk-1].get_xaxis().set_visible(False)
        axs[jj, kk-1].get_yaxis().set_visible(False)
#        titleString = str(jj)+' vs '+str(kk)
#        axs[jj, kk].set_title('Axis [0, 0]')
plt.show()
# In[65]:

# compute correlations:
# Compute vector inner product between columns by taking X^T @X (with nan's replaced by 0's)
# Find number of nontrivial products by taking Xan^T @ Xan
# Pointwise divide to find correlations
Xcov = (X0.transpose()@ X0) / (Xan.transpose() @ Xan)
print(Xcov)
# Find eigenvectors and eigenvalues
lam,eVec = LA.eigh(Xcov)

# Show seasonality
plt.scatter(np.arange(len(X))%48,X[:,0],c = np.arange(len(X))%(48*365), s = 1)
plt.colorbar()
plt.title("Daily and yearly seasonality of FEE")
plt.xlabel("Half-hour time stamp")
plt.ylabel("Normalized value of FEE")
plt.show()

ndp = 3
Xyr = np.reshape(X[:,0],(ndp,48//ndp,-1))
Xyr = np.nanmean(Xyr,axis=0)
nh,ny = np.shape(Xyr)
yrInd = np.ones((nh,1))@np.reshape(np.arange(ny),(1,-1))
yrInd = yrInd.astype(int) % 365
hInd = np.reshape(np.arange(nh),(-1,1))@np.ones((1,ny))
for jh in range(nh):
    plt.plot(yrInd[jh,:],savgol_filter(Xyr[jh,:],7,2),'-')
plt.title('Time of day variation over the year')
plt.show()

#@@@ Find distribution of hole sizes.

#@@@ Find distribution of non-hole sizes.
wXnan = np.where(Xnan[:,0])[0]
wXnanDiff = wXnan[1:]-wXnan[:-1]-1
wXnanDiff = wXnanDiff[wXnanDiff>0]
wXnanDiff = wXnanDiff[wXnanDiff<50]
plt.hist(wXnanDiff,bins=50)
plt.title("Distribution of available FEE data intervals")
plt.show()

wXan = np.where(Xan[:,0])[0]
wXanDiff = wXan[1:]-wXan[:-1]-1
wXanDiff = wXanDiff[wXanDiff>0]
wXanDiff = wXanDiff[wXanDiff<50]
plt.hist(wXanDiff,bins=50)
plt.title("Distribution of nan FEE data runs")
plt.show()



