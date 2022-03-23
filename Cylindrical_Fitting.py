import numpy as np
from scipy import optimize
from scipy.optimize import leastsq
from scipy.optimize import minimize
import pandas as pd
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math

#-*- coding: utf-8 -*-
myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

def xclinerrors(para, points): 
    """法向量x矢量最大，圆柱面拟合误差"""
    # points为由点集构成的n*3矩阵，每一行为一个点的x,y,z坐标值
    x0 = 0
    y0, z0, a0, b0, c0, r0 = para
    return (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + (points[:, 2] - z0) ** 2 - (
            a0 * (points[:, 0] - x0) + b0 * (points[:, 1] - y0) + c0 * (points[:, 2] - z0)) ** 2 - r0 ** 2


def yclinerrors(para, points):
    """法向量y矢量最大，圆柱面拟合误差"""
    y0 = 0
    x0, z0, a0, b0, c0, r0 = para
    return (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + (points[:, 2] - z0) ** 2 - (
            a0 * (points[:, 0] - x0) + b0 * (points[:, 1] - y0) + c0 * (points[:, 2] - z0)) ** 2 - r0 ** 2


def zclinerrors(para, points):
    """法向量z矢量最大，圆柱面拟合误差"""
    z0 = 0
    x0, y0, a0, b0, c0, r0 = para
    return (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2 + (points[:, 2] - z0) ** 2 - (
            a0 * (points[:, 0] - x0) + b0 * (points[:, 1] - y0) + c0 * (points[:, 2] - z0)) ** 2 - r0 ** 2

def direction(theta, phi):
    '''Return the direction vector of a cylinder defined
    by the spherical coordinates theta and phi.
    '''
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                     np.cos(theta)])

def projection_matrix(w):
    '''Return the projection matrix  of a direction w.'''
    return np.identity(3) - np.dot(np.reshape(w, (3,1)), np.reshape(w, (1, 3)))

def skew_matrix(w):
    '''Return the skew matrix of a direction w.'''
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def calc_A(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum(np.dot(np.reshape(Y, (3,1)), np.reshape(Y, (1, 3)))
            for Y in Ys)

def calc_A_hat(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return np.dot(S, np.dot(A, np.transpose(S)))

def preprocess_data(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean

def G(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    
    u = sum(np.dot(Y, Y) for Y in Ys) / n
    v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

    return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)

def C(w, Xs):
    '''Calculate the cylinder center given the cylinder direction and 
    a list of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))

def r(w, Xs):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    c = C(w, Xs)

    return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)

def fit(data, guess_angles=None):
    '''Fit a list of data points to a cylinder surface. The algorithm implemented
    here is from David Eberly's paper "Fitting 3D Data with a Cylinder" from 
    https://www.geometrictools.com/Documentation/CylinderFitting.pdf

    Arguments:
        data - A list of 3D data points to be fitted.
        guess_angles[0] - Guess of the theta angle of the axis direction
        guess_angles[1] - Guess of the phi angle of the axis direction
    
    Return:
        Direction of the cylinder axis
        A point on the cylinder axis
        Radius of the cylinder
        Fitting error (G function)
    '''
    Xs, t = preprocess_data(data)  

    # Set the start points

    start_points = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    if guess_angles:
        start_points = guess_angles

    # Fit the cylinder from different start points 

    best_fit = None
    best_score = float('inf')

    for sp in start_points:
        fitted = minimize(lambda x : G(direction(x[0], x[1]), Xs),
                    sp, method='Powell', tol=1e-6)

        if fitted.fun < best_score:
            best_score = fitted.fun
            best_fit = fitted

    w = direction(best_fit.x[0], best_fit.x[1])

    return w, C(w, Xs) + t, r(w, Xs), best_fit.fun 

def rotation_matrix_from_axis_and_angle(u, theta):
    '''Calculate a rotation matrix from an axis and an angle.'''
    
    x = u[0]
    y = u[1]
    z = u[2]
    s = np.sin(theta)
    c = np.cos(theta)
    
    return np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y**2 * (1 - c), y * z * (1 - c) - x * s ],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c) ]])

# 读取数据
UL_step = []
theta = []
retheta = [6]*200  ##10可替换成超声波阵列采样次数
new_ULx = []
new_ULy = []
ts = []

dfx = pd.read_excel('bottle(25mm)_1.xlsx',usecols=[1],names=None)
dfx = dfx.values.tolist() 
for i in dfx:
    UL_step.append(i[0])

dft = pd.read_excel('bottle_time(25mm)_1.xlsx',usecols=[1],names=None)
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

points = np.append(xys, zs, axis=1)




 # 计算点集的x,y,z均值，x，y的范围，用于拟合参数初值
xm = np.mean(points[:,0])    # 特征点集x均值
ym = np.mean(points[:,1])
zm = np.mean(points[:,2])         
xrange = np.max(points[:,0]) - np.min(points[:,0])    # 特征点集x范围
yrange = np.max(points[:,1]) - np.min(points[:,1])



cnv, C_fit, r_fit, fit_err = fit(points)

print('中心点%s 圆柱法向量%s 半径%s' %(cnv, C_fit, r_fit))


# 最小二乘拟合 
# x分量最大，则法向量必然与yz平面相交，轴点初值中x可设为0
# cnv为公法线
if cnv[0] >= cnv[1] and cnv[0] >= cnv[2]:   
    tparac = leastsq(xclinerrors, [ym, zm, cnv[0], cnv[1], cnv[2], yrange/2], points,
                                              full_output=1)
    parac = np.array([0, tparac[0][0], tparac[0][1], tparac[0][2], tparac[0][3], tparac[0][4], tparac[0][5]])
    # 圆度误差均值
    ec = np.mean(np.abs(tparac[2]['fvec'])) / parac[6]
    
if cnv[1] >= cnv[0] and cnv[1] >= cnv[2]:
    tparac = leastsq(yclinerrors, [xm, zm, cnv[0], cnv[1], cnv[2], xrange/2], points,
                                              full_output=1)
    parac = np.array([tparac[0][0], 0, tparac[0][1], tparac[0][2], tparac[0][3], tparac[0][4], tparac[0][5]])
    ec = np.mean(np.abs(tparac[2]['fvec'])) / parac[6]
    
if cnv[2] >= cnv[0] and cnv[2] >= cnv[1]:
    tparac = leastsq(zclinerrors, [xm, ym, cnv[0], cnv[1], cnv[2], (xrange + yrange) / 2],
                                             points,full_output=1)
    parac = np.array([tparac[0][0], tparac[0][1], 0, tparac[0][2], tparac[0][3], tparac[0][4], tparac[0][5]])
    ec = np.mean(np.abs(tparac[2]['fvec'])) / parac[6]



fig = plt.figure()
ax = fig.gca(projection="3d")

ax.scatter(points.T[0],points.T[1],points.T[2],zdir="z",marker="o",s=4)


ax.set(xlabel="X",ylabel="Y",zlabel="Z")
ax.set_title(u'柱面点云及最小二乘拟合图',fontproperties=myfont)




# Get the transformation matrix

theta = np.arccos(np.dot(cnv, np.array([0, 0, 1])))
phi = np.arctan2(cnv[1], cnv[0])

M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
            rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

# Plot the cylinder surface

delta = np.linspace(-np.pi, np.pi, 200)
z = np.linspace(-10, 10, 200)

Delta, Z = np.meshgrid(delta, z)
X = r_fit * np.cos(Delta)
Y = r_fit * np.sin(Delta)

for i in range(len(X)):
    for j in range(len(X[i])):
        p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + C_fit

        X[i][j] = p[0]
        Y[i][j] = p[1]
        Z[i][j] = p[2]

ax.plot_wireframe(X, Y, Z, color=(0, 1, 0, 0.5),rstride=10, cstride=10)

plt.show()