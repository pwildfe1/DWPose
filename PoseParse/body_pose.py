import cv2
import os
import shutil
import math as m
import numpy as np
import matplotlib.pyplot as plt



class PoseSet:


    def __init__(self, array):

        self.pts = array

        self.profile = self.pts[0] - self.pts[3]
        self.orientation = np.dot(self.profile, [1, 0])/abs(np.dot(self.profile, [1, 0])) * np.array([1, 0, 0])
        self.shoulders = self.pts[6] - self.pts[5]
        self.hip = self.pts[12] - self.pts[11]
        self.thigh01 = self.pts[13] - self.pts[11]
        self.thigh02 = self.pts[14] - self.pts[12]
        self.calf01 = self.pts[15] - self.pts[13]
        self.calf02 = self.pts[16] - self.pts[14]

        self.hip_center = (self.pts[12] + self.pts[11]) / 2
        self.torso_center = (self.pts[6] + self.pts[5] + self.pts[12] + self.pts[11])/4
        self.torso_align = self.pts[11] - self.pts[5]
        self.torso_align = self.torso_align/np.linalg.norm(self.torso_align)

        self.shoulder_width, self.hip_width = 0, 0
        self.forearm_height, self.upperarm_height, self.torso_height, self.thigh_height, self.calf_height = 0, 0, 0, 0, 0

        self.nodes = np.zeros((12, 3))


    def add_measurements(self, measurements):

        self.shoulder_width = measurements["shoulder width"]
        self.hip_width = measurements["hip width"]

        self.forearm_height = measurements["forearm height"]
        self.upperarm_height = measurements["upperarm height"]
        self.torso_height = measurements["torso height"]
        self.thigh_height = measurements["thigh height"]
        self.calf_height = measurements["calf height"]


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