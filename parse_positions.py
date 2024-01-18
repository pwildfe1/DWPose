import cv2
import os
import shutil
import math as m
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


"""
5->6 (shoulders)

11->12 (hips)

11->13 (thigh 1)

12->14 (thigh 2)

13->15 (calf 1)

14->16 (calf 2)
"""

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





def main(source_folder, dest_folder):

    count = 0

    for file in os.listdir(source_folder):

        if ".npy" in file:
            pts = np.load(source_folder + "/" + file)
            if pts.shape[0] == 17:
                pose = PoseSet(pts)
                ang_01, ang_02 = pose.calculate_leg_bend()
                loading_ratio = pose.evaluate_loading()
                if ang_01 < m.pi/3 and ang_02 < m.pi/3 and np.linalg.norm(pose.hip) < 100 and loading_ratio < 1 and pose.torso_align[1] > .6:
                    img = Image.open(source_folder + "/" + file.split(".")[0] + ".jpg")
                    im = np.array(img)
                    im[:,:,0] = 255
                    img = Image.fromarray(im)
                    img.save(dest_folder + "/" + file.split(".")[0] + ".jpg")
                    count = count+1
                    # for i in range(5):
                    #     img.save(dest_folder + "/frame" + str(count) + ".jpg")
                    #     count = count + 1
                else:
                    shutil.copyfile(source_folder + "/" + file.split(".")[0] + ".jpg", dest_folder + "/" + file.split(".")[0] + ".jpg")
                    count = count + 1



main("out_image", "select_image")