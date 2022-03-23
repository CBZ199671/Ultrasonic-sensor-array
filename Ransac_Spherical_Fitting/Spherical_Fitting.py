import numpy as np
import random
import copy 
import pandas as pd
import math
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#-*- coding: utf-8 -*-
myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

def get_rotationMatrix_from_vectors(u, v):
    """ 
    Create a rotation matrix that rotates the space from a 3D vector `u` to a 3D vector `v`

    :param u: Orign vector `np.array (1,3)`.
    :param v: Destiny vector `np.array (1,3)`.

    :returns: Rotation matrix `np.array (3, 3)`

    ---
    """

    # Lets find a vector which is ortogonal to both u and v
    w = np.cross(u, v)
    
    # This orthogonal vector w has some interesting proprieties
    # |w| = sin of the required rotation 
    # dot product of w and goal_normal_plane is the cos of the angle
    c = np.dot(u, v)
    s = np.linalg.norm(w)


    # Now, we compute rotation matrix from rodrigues formula 
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    # We calculate the skew symetric matrix of the ort_vec
    Sx = np.asarray([[0, -w[2], w[1]],
                   [w[2], 0, -w[0] ],
                   [-w[1], w[0], 0]])
    R = np.eye(3) + Sx + Sx.dot(Sx) * ((1 - c) / (s ** 2))
    return R



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

class Sphere:
    """ 
    Implementation for Sphere RANSAC. A Sphere is defined as points spaced from the center by a constant radius. 


    This class finds the center and radius of a sphere. Base on article "PGP2X: Principal Geometric Primitives Parameters Extraction"

    ![3D Sphere](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/sphere.gif "3D Sphere")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.center = []
        self.radius = 0

    def fit(self, pts, thresh=0.2, maxIteration=1200):
        """ 
        Find the parameters (center and radius) to define a Sphere. 

        :param pts: 3D point cloud as a numpy array (N,3).
        :param thresh: Threshold distance from the Sphere hull which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.

        :returns: 
        - `center`: Center of the cylinder np.array(1,3) which the cylinder axis is passing through.
        - `radius`: Radius of cylinder.
        - `inliers`: Inlier's index from the original point cloud.
        ---
        """

        n_points = pts.shape[0]
        best_inliers = []

        for it in range(maxIteration):

            # Samples 4 random points 
            id_samples = random.sample(range(1, n_points-1), 4)
            pt_samples = pts[id_samples]

            # We calculate the 4x4 determinant by dividing the problem in determinants of 3x3 matrix

            # Multiplied by (x²+y²+z²)
            d_matrix = np.ones((4, 4))
            for i in range(4):
                d_matrix[i, 0] = pt_samples[i, 0]
                d_matrix[i, 1] = pt_samples[i, 1]
                d_matrix[i, 2] = pt_samples[i, 2]
            M11 = np.linalg.det(d_matrix)

            # Multiplied by x
            for i in range(4):
                d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
                d_matrix[i, 1] = pt_samples[i, 1]
                d_matrix[i, 2] = pt_samples[i, 2]
            M12 = np.linalg.det(d_matrix)

            # Multiplied by y
            for i in range(4):
                d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
                d_matrix[i, 1] = pt_samples[i, 0]
                d_matrix[i, 2] = pt_samples[i, 2]
            M13 = np.linalg.det(d_matrix)

            # Multiplied by z
            for i in range(4):
                d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
                d_matrix[i, 1] = pt_samples[i, 0]
                d_matrix[i, 2] = pt_samples[i, 1]
            M14 = np.linalg.det(d_matrix)


            # Multiplied by 1
            for i in range(4):
                d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
                d_matrix[i, 1] = pt_samples[i, 0]
                d_matrix[i, 2] = pt_samples[i, 1]
                d_matrix[i, 3] = pt_samples[i, 2]
            M15 = np.linalg.det(d_matrix)

            # Now we calculate the center and radius
            center = [0.5*(M12/M11), -0.5*(M13/M11), 0.5*(M14/M11)]
            radius = np.sqrt(np.dot(center, center) - (M15 / M11))


            # Distance from a point
            pt_id_inliers = [] # list of inliers ids
            dist_pt = center - pts
            dist_pt = np.linalg.norm(dist_pt, axis=1)


            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt-radius) <= thresh)[0]

            if(len(pt_id_inliers) > len(best_inliers)):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.center = center
                self.radius = radius

        return self.center, self.radius,  self.inliers

# 读取数据
UL_step = []
theta = []
retheta = [6]*200  ##10可替换成超声波阵列采样次数
new_ULx = []
new_ULy = []
ts = []

dfx = pd.read_excel('youzi.xlsx',usecols=[1],names=None)
dfx = dfx.values.tolist() 
for i in dfx:
    UL_step.append(i[0])

dft = pd.read_excel('youzi_time.xlsx',usecols=[1],names=None)
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



points = xyzs

sph = Sphere()

center, radius, inliers = sph.fit(points, thresh=0.4)
print("center: "+str(center))
print("radius: "+str(radius))

u = np.linspace(0, 2 * np.pi, 200)
v = np.linspace(0, np.pi, 200)
x_p = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
y_p = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
z_p = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

fig = plt.figure()
ax = fig.gca(projection="3d")

ax.scatter(xyzs.T[0],xyzs.T[1],xyzs.T[2],zdir="z",marker="o",s=4)


ax.set(xlabel="X",ylabel="Y",zlabel="Z")
ax.set_title(u'球面点云及Ransac拟合图',fontproperties=myfont)

ax.plot_wireframe(x_p, y_p, z_p, color=(0, 1, 0, 0.5),rstride=10, cstride=10)
ax.set_zlim([center[2]-radius,center[2]+radius])
plt.show()