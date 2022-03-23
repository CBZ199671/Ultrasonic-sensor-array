import numpy as np
import pandas as pd
from numpy.core.fromnumeric import reshape

df = pd.read_excel('suface.xlsx',usecols=[1],names=None)
df = df.values.tolist() 
x = []
for j in df:
    x.append(j[0])
x = np.array(x)
x = x.reshape(-1,1)

y = np.arange(0,10,1)
y = np.repeat(y,150) 
y = y.reshape(-1,1)

xys = np.append(x, y, axis=1)

z = np.arange(0,13,2.4)
z = np.tile(z,250) 
z = z.reshape(-1,1) 

xyzs = np.append(xys, z, axis=1)

print(xyzs)
