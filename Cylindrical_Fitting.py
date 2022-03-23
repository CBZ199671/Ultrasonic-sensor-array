import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math
import random

#-*- coding: utf-8 -*-
myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

def rodrigues_rot(P, n0, n1):
    """ 
    Rotate a set of point between two normal vectors using Rodrigues' formula. 

    :param P: Set of points `np.array (N,3)`.
    :param n0: Orign vector `np.array (1,3)`.
    :param n1: Destiny vector `np.array (1,3)`.

    :returns: Set of points P, but rotated `np.array (N, 3)`

    ---
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    P_rot = np.zeros((len(P),3))
    if(np.linalg.norm(k)!=0):
        k = k/np.linalg.norm(k)
        theta = np.arccos(np.dot(n0,n1))
        
        # Compute rotated points
        for i in range(len(P)):
            P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))
    else:
        P_rot = P
    return P_rot

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


class Cylinder:
    """ 
    !!! warning
        The cylinder RANSAC in this library works! but does not present good results on real data on the current version.
        We are working to make a better algorithim using normals. If you want to contribute, please create a MR on github.
        You'll be our hero! 

    Implementation for cylinder RANSAC.

    This class finds a infinite height cilinder and returns the cylinder axis, center and radius. 

    ---
    """
    

    def __init__(self):
        self.inliers = []
        self.center = []
        self.axis = []
        self.radius = 0

    def fit(self, pts, thresh=0.2, maxIteration=10000):
        """ 
        Find the parameters (axis and radius) defining a cylinder. 

        :param pts: 3D point cloud as a numpy array (N,3).
        :param thresh: Threshold distance from the cylinder hull which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.

        :returns: 
        - `center`: Center of the cylinder np.array(1,3) which the cylinder axis is passing through.
        - `axis`: Vector describing cylinder's axis np.array(1,3).
        - `radius`: Radius of cylinder.
        - `inliers`: Inlier's index from the original point cloud.
        ---
        """

        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in range(maxIteration):

            # Samples 3 random points 
            id_samples = random.sample(range(1, n_points-1), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1,:] - pt_samples[0,:]
            vecA_norm = vecA / np.linalg.norm(vecA)
            vecB = pt_samples[2,:] - pt_samples[0,:]
            vecB_norm = vecB / np.linalg.norm(vecB)

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA_norm, vecB_norm)
            vecC = vecC / np.linalg.norm(vecC)

            # Now we calculate the rotation of the points with rodrigues equation
            P_rot = rodrigues_rot(pt_samples, vecC, [0,0,1])

            # Find center from 3 points
            # http://paulbourke.net/geometry/circlesphere/
            # Find lines that intersect the points
            # Slope:
            ma = 0
            mb = 0
            while(ma == 0):
                ma = (P_rot[1, 1]-P_rot[0, 1])/(P_rot[1, 0]-P_rot[0, 0])
                mb = (P_rot[2, 1]-P_rot[1, 1])/(P_rot[2, 0]-P_rot[1, 0])
                if(ma == 0):
                    P_rot = np.roll(P_rot,-1,axis=0)
                else:
                    break

            # Calulate the center by verifying intersection of each orthogonal line
            p_center_x = (ma*mb*(P_rot[0, 1]-P_rot[2, 1]) + mb*(P_rot[0, 0]+P_rot[1, 0]) - ma*(P_rot[1, 0]+P_rot[2, 0]))/(2*(mb-ma))
            p_center_y = -1/(ma)*(p_center_x - (P_rot[0, 0]+P_rot[1, 0])/2)+(P_rot[0, 1]+P_rot[1, 1])/2
            p_center = [p_center_x, p_center_y, 0]
            radius = np.linalg.norm(p_center - P_rot[0, :])

            # Remake rodrigues rotation
            center = rodrigues_rot(p_center, [0,0,1], vecC)[0]

            # Distance from a point to a line
            pt_id_inliers = [] # list of inliers ids
            vecC_stakado =  np.stack([vecC]*n_points,0)
            dist_pt = np.cross(vecC_stakado, (center- pts))
            dist_pt = np.linalg.norm(dist_pt, axis=1)


            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt-radius) <= thresh)[0]

            if(len(pt_id_inliers) > len(best_inliers)):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.center = center
                self.axis = vecC
                self.radius = radius

        return self.center, self.axis, self.radius,  self.inliers


# 读取数据
UL_step = []
theta = []
retheta = [6]*200  ##10可替换成超声波阵列采样次数
new_ULx = []
new_ULy = []
ts = []

dfx = pd.read_excel('bottle(300mm)_10.xlsx',usecols=[1],names=None)
dfx = dfx.values.tolist() 
for i in dfx:
    UL_step.append(i[0])

dft = pd.read_excel('bottle_time(300mm)_10.xlsx',usecols=[1],names=None)
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



cil = Cylinder()

center, normal, radius,  inliers = cil.fit(points, thresh=0.05)
print("center: "+str(center))
print("radius: "+str(radius))
print("vecC: "+str(normal))

fig = plt.figure()
ax = fig.gca(projection="3d")

ax.scatter(points.T[0],points.T[1],points.T[2],zdir="z",marker="o",s=4)


ax.set(xlabel="X",ylabel="Y",zlabel="Z")
ax.set_title(u'柱面点云及Ransac拟合图',fontproperties=myfont)

# Get the transformation matrix

theta = np.arccos(np.dot(normal, np.array([0, 0, 1])))
phi = np.arctan2(normal[1], normal[0])

M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
            rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), theta))

# Plot the cylinder surface

delta = np.linspace(-np.pi, np.pi, 200)
z = np.linspace(0, 12, 200)

Delta, Z = np.meshgrid(delta, z)
X = radius * np.cos(Delta)
Y = radius * np.sin(Delta)

for i in range(len(X)):
    for j in range(len(X[i])):
        p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]])) + center

        X[i][j] = p[0]
        Y[i][j] = p[1]
        Z[i][j] = p[2]

ax.plot_wireframe(X, Y, Z, color=(0, 1, 0, 0.5),rstride=10, cstride=10)

plt.show()