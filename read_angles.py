import cv2
import os
import shutil
import math as m
import numpy as np
import matplotlib.pyplot as plt


class PoseSet:


    def __init__(self, array):

        self.pts = array

        self.shoulders = self.pts[6] - self.pts[5]
        self.hip = self.pts[12] - self.pts[11]
        self.thigh01 = self.pts[13] - self.pts[11]
        self.thigh02 = self.pts[14] - self.pts[12]
        self.calf01 = self.pts[15] - self.pts[13]
        self.calf02 = self.pts[16] - self.pts[14]

        self.torso_center = (self.pts[6] + self.pts[5] + self.pts[12] + self.pts[11])/4
        self.torso_align = self.pts[11] - self.pts[5]
        self.torso_align = self.torso_align/np.linalg.norm(self.torso_align)



    def calculate_leg_bend(self):

        calf_01 = -self.calf01/(np.linalg.norm(self.calf01))
        thigh_01 = self.thigh01/(np.linalg.norm(self.thigh01))

        ang_01 = m.acos(np.dot(calf_01, thigh_01))


        calf_02 = -self.calf01/(np.linalg.norm(self.calf01))
        thigh_02 = self.thigh01/(np.linalg.norm(self.thigh01))

        ang_02 = m.acos(np.dot(calf_01, thigh_01))

        return [ang_01, ang_02]


    def calculate_point_overload(self, pt):
        vec = pt - (self.pts[5] + self.pts[6])/2
        return vec[0]


    def evaluate_loading(self):

        load_ankle_01 = self.calculate_point_overload(self.pts[15])
        load_ankle_02 = self.calculate_point_overload(self.pts[16])
        load_ankle = np.min([load_ankle_01, load_ankle_02])


        load_knee_01 = self.calculate_point_overload(self.pts[13])
        load_knee_02 = self.calculate_point_overload(self.pts[14])
        load_knee = np.min([load_knee_01, load_knee_02])

        return load_ankle/load_knee



def extract_leg_angles(side, array, height, weight):

    pose = PoseSet(array)

    if side == 0:
        hip = pose.pts[11]
        knee = pose.pts[13]
        ankle = pose.pts[15]
    else:
        hip = pose.pts[12]
        knee = pose.pts[14]
        ankle = pose.pts[16]

    thigh = hip - knee
    thigh = thigh/np.linalg.norm(thigh)

    calf = ankle - knee
    calf = calf/np.linalg.norm(calf)

    hip_incline = m.acos(np.dot(-thigh, [-thigh[0]/abs(thigh[0]), 0]))
    leg_bend = m.acos(np.dot(calf, thigh))

    print(hip_incline)
    print(leg_bend)




array = np.load("out_image/frame667.npy")
extract_leg_angles(0, array, 6, 195)