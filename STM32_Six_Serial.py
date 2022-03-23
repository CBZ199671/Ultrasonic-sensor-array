
from matplotlib.pyplot import xlabel, ylabel
import serial
import time
import numpy as np 
from numpy.lib.shape_base import tile 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import re
import math

#连接串口
serial = serial.Serial('COM3',115200,timeout=2)
UL_step = []
t = []
theta = []
new_ULx = []
new_ULy = []
retheta = [6]*200  ##10可替换成超声波阵列采样次数

if serial.isOpen():
    print('串口已打开')
    for i in range(200): ##超声波阵列采样次数
        t.append(time.time())
        for j in range(6):  ##超声波个数
            data = str(serial.readline())
            m = re.search('[\d\.]+', data)
            renewdata = 15 - float(m.group())  ##超声波阵列中轴和转台中心的距离单位（cm）
            UL_step.append(renewdata)    
else:
    print ('串口未打开')

#关闭串口
serial.close()
 
if serial.isOpen():
	print ('串口未关闭')
else:
	print ('串口已关闭')

#求出每次采样间隔角度
theta.append(27.5*(t[1]-t[0]))
for i in range(198):     ##比超声波采样次数少2
    theta.append(27.5*(t[i+2]-t[i+1]))
#求解每次采样的角度
s = np.sum(theta)
retheta[0] = s
for i in range(198):     #比超声波阵列采样次数少2
    retheta[i+1] = retheta[i] - theta[i]

retheta = np.repeat(retheta,6) #超声波个数

""" a = [3,4,3,6,7]
b = [0]*5
s = np.sum(a)
b[0] = s
for i in range(4):
    b[i+1] = b[i] - a[i] 
print(b) """


for j in range(1200):  ##超声波采样总个数
    new_ULx.append(UL_step[j]*math.cos(math.radians(retheta[j])))
    new_ULy.append(UL_step[j]*math.sin(math.radians(retheta[j])))

z1 = np.arange(0,18,3.5)
z1 = np.tile(z1,200)  ##超声波阵列采样次数


fig = plt.figure()
ax = fig.gca(projection="3d")


ax.scatter(new_ULx,new_ULy,z1,zdir="z",c="purple",marker="o",s=4)

ax.set(xlabel="X",ylabel="Y",zlabel="Z")

plt.show()

df = pd.DataFrame(UL_step)
df.to_excel('C:\\Users\\huolo\\Desktop\\test\\youzi.xlsx')
df = pd.DataFrame(t)
df.to_excel('C:\\Users\\huolo\\Desktop\\test\\youzi_time.xlsx')