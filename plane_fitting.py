import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

#-*- coding: utf-8 -*-
myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)



def planeerrors(para, xyzs):
    """平面误差"""
    a0, a1, a2 = para
    return a0 *  xyzs[:, 1] + a1 * xyzs[:, 2] + a2 - xyzs[:, 0]


# read data
df = pd.read_excel('PingMianTest(20mm).xlsx',usecols=[1],names=None)
df = df.values.tolist() 
xs = []
for j in df:
    xs.append(j[0])
xs = np.array(xs)
xs = xs.reshape(-1,1)

ys = np.arange(0,10,1)
ys = np.repeat(ys,6)
ys = ys.reshape(-1,1)

xys = np.append(xs, ys, axis=1)

zs = np.arange(0,12,2.3)
zs = np.tile(zs,10)  
zs = zs.reshape(-1,1)

xyzs = np.append(xys, zs, axis=1)

""" #立方体测试
xyzs = [[2,0,0],[2, 2, 0],[2, 2, 2],[2, 0, 2]]
#[0, 0, 0],[0, 2, 0],[2, 2, 0],[2, 0, 0],[0, 0, 2],[0, 2, 2],[2, 2, 2],[2, 0, 2]
xyzs = np.array(xyzs)
print(xyzs) """

fig = plt.figure()
ax = fig.gca(projection="3d")

ax.scatter(xyzs.T[0],xyzs.T[1],xyzs.T[2],zdir="z",marker="o",s=40)
print(xyzs.T[0])

ax.set(xlabel="X",ylabel="Y",zlabel="Z")
ax.set_title(u'平面点云及拟合图',fontproperties=myfont) 


    
tparap = leastsq(planeerrors, [0, 0, 0], xyzs)
para = tparap[0]
print(para)
y_p = np.linspace(0,10,100)
z_p = np.linspace(0,12,100)
y_p,z_p = np.meshgrid(y_p, z_p)
x_p = para[0]*y_p + para[1]*z_p + para[2]
print(x_p)
ax.plot_wireframe(x_p, y_p, z_p, color=(0, 1, 0, 0.5),rstride=10, cstride=10)


plt.show()