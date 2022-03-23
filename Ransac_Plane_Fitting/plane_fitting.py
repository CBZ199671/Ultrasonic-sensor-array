import numpy as np
from ransac import *
import pandas as pd

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

if __name__ == '__main__':
    from matplotlib.pyplot import title, xlabel, ylabel
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
        #-*- coding: utf-8 -*-
    myfont = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    def plot_plane(a, b, c, d):
        yy, zz = np.mgrid[:10, :14]
        return (-b*yy-c*zz-d)/a, yy, zz

    n = 60
    max_iterations = 60
    goal_inliers = n * 0.3

    # read data
    df = pd.read_excel('PingMianTest(300mm).xlsx',usecols=[1],names=None)
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

    ax.scatter(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])
    ax.set_title(u'平面点云及拟合图',fontproperties = myfont) 
    ax.set(xlabel="X",ylabel="Y",zlabel="Z")
    

    # RANSAC
    m, b = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.01), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    xx, yy, zz = plot_plane(a, b, c, d)
    ax.plot_wireframe(xx, yy, zz, color=(0, 1, 0))
    plt.show()
    
