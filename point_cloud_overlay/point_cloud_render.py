import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from scipy import interpolate
from scipy.spatial.transform import Rotation


class PinholeCamera:

    def __init__(self, fx, fy, px = 0, py = 0):
        self.fx, self.fy = fx, fy
        self.px, self.py = px, py
        self.K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
        self.I = np.identity(3)
        self.I = np.vstack((self.I, np.zeros((3)))).T
        self.P = np.matmul(self.K, self.I)

    def setCameraTransform(self, position, angles):
        self.I = Rotation.from_euler('xyz', angles, degrees = True).as_matrix()
        position = np.array(position)
        self.I = np.vstack((self.I, position)).T
        self.P = np.matmul(self.K, self.I)
        print(self.I)
        print(self.P)

    def renderPts(self, pts, colors):
        px = []
        for v in pts:
            pt = np.zeros((4,1))
            pt[0] = v[0]
            pt[1] = v[1]
            pt[2] = v[2]
            pt[3] = 1
            homogenous = np.matmul(self.P,pt)
            px.append([self.fx * homogenous[0]/homogenous[2] + self.px, self.fy * homogenous[1]/homogenous[2] + self.py])
        px = np.array(px)
        print(px[0:10])
        fig, ax = plt.subplots(1, 1, figsize=(10,10))
        ax.scatter(px[:,0], px[:,1], color=colors, s=1)
        ax.set_ylim(-940, 940)
        ax.set_xlim(-940, 940)
        plt.show()

        print(np.max(pts[:,2]) - np.min(pts[:, 2]))
        print(np.max(px[:,2]) - np.min(px[:, 2]))


def main(file):
    pcd = o3d.io.read_point_cloud(file)
    pcd = pcd.scale(100, [0, 0, 0])
    v = np.asarray(pcd.points)
    pcd = pcd.translate([0, 0, -np.max(v[:,2]) - 50])
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.001, max_nn=30)
    myCamera = PinholeCamera(35, 35)
    myCamera.setCameraTransform([0, 0, 0], [0, 0, 0])

    v_colors = np.asarray(pcd.colors)

    v = np.asarray(pcd.points)
    myCamera.renderPts(v, v_colors)

    # o3d.visualization.draw_geometries([pcd])


main("pc.ply")