import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math

#-*- coding: utf-8 -*-
myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)


def spherrors(para, xyzs):
    """球面拟合误差"""
    x0, y0, z0, r0 = para
    return (xyzs[:, 0] - x0) ** 2 + (xyzs[:, 1] - y0) ** 2 + (xyzs[:, 2] - z0) ** 2 - r0 ** 2


# 读取数据
UL_step = []
theta = []
retheta = [6]*200  ##10可替换成超声波阵列采样次数
new_ULx = []
new_ULy = []
ts = []

dfx = pd.read_excel('football(300mm)_10.xlsx',usecols=[1],names=None)
dfx = dfx.values.tolist() 
for i in dfx:
    UL_step.append(i[0])

dft = pd.read_excel('football_time(300mm)_10.xlsx',usecols=[1],names=None)
dft = dft.values.tolist() 
for j in dft:
    ts.append(j[0])


#求出每次采样间隔角度
theta.append(27.5*(ts[1]-ts[0]))
for i in range(198):     ##比超声波采样次数少2
    theta.append(27.5*(ts[i+2]-ts[i+1]))

#求解每次采样的角度
s = np.sum(theta)
retheta[0] = s
for i in range(198):     #比超声波阵列采样次数少2
    retheta[i+1] = retheta[i] - theta[i]

retheta = np.repeat(retheta,6) #超声波个数

for j in range(1200):  ##超声波采样总个数
    new_ULx.append(UL_step[j]*math.cos(math.radians(retheta[j])))
    new_ULy.append(UL_step[j]*math.sin(math.radians(retheta[j])))
xs = np.array(new_ULx)
xs = xs.reshape(-1,1)

ys = np.array(new_ULy)
ys = ys.reshape(-1,1)

xys = np.append(xs, ys, axis=1)

zs = np.arange(0,12,2.3)
zs = np.tile(zs,200)  
zs = zs.reshape(-1,1)

xyzs = np.append(xys, zs, axis=1)






# 计算点集的x,y,z均值，用于拟合参数初值
xm = np.mean(xyzs[:,0])    # 特征点集x均值
ym = np.mean(xyzs[:,1])
zm = np.mean(xyzs[:,2])
# 最小二乘拟合
tparas = leastsq(spherrors, [xm, ym, zm, 0.001], xyzs, full_output=1)
paras = tparas[0]
print(paras)
# 计算球度误差
es = np.mean(np.abs(tparas[2]['fvec'])) / paras[3]    # 'fvec'即为spherrors的值




fig = plt.figure()
ax = fig.gca(projection="3d")

ax.scatter(xyzs.T[0],xyzs.T[1],xyzs.T[2],zdir="z",marker="o",s=4)


ax.set(xlabel="X",ylabel="Y",zlabel="Z")
ax.set_title(u'球面点云及最小二乘拟合图',fontproperties=myfont)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_p = paras[3] * np.outer(np.cos(u), np.sin(v)) + paras[0]
y_p = paras[3] * np.outer(np.sin(u), np.sin(v)) + paras[1]
z_p = paras[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + paras[2]

ax.plot_wireframe(x_p, y_p, z_p, color=(0, 1, 0, 0.5),rstride=10, cstride=10)

#ax.set_xlim([paras[0]-paras[3],paras[0]+paras[3]])

#ax.set_ylim([paras[1]-paras[3],paras[1]+paras[3]])

ax.set_zlim([paras[2]-paras[3],paras[2]+paras[3]])

plt.show()