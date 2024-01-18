import os
import logging
import json
import math as m
import numpy as np
import matplotlib.pyplot as plt

import PoseParse.body_pose as BDPose
import PinholeCamera.pinhole_camera as Camera

# from matplotlib.patches import Circle
from PIL import Image
# from mmpose.apis import inference_topdown, init_model
# from mmpose.utils import register_all_modules

# from mmcv.image import imread
# from mmengine.logging import print_log

# from mmpose.apis import inference_topdown, init_model
# from mmpose.registry import VISUALIZERS
# from mmpose.structures import merge_data_samples

# register_all_modules()


def apply_measurement(camera, pose, pixel, ref_pt, measurement, direction = 1.0):

    # pt = camera.calculate_xyz(pixel, ref_pt[2])
    pt = [pixel[0] - ref_pt[0], pixel[1] - ref_pt[1], 0]

    z = m.pow(m.pow(measurement,2) - m.pow(pt[0],2) - m.pow(pt[1], 2),.5)

    # pt = [pixel[0], pixel[1], ref_pt[2] + z * direction]

    pt = [pixel[0], pixel[1], ref_pt[2] + z * direction]

    return pt

"""
11->12 (hips)

11->13 (thigh 1)

12->14 (thigh 2)

13->15 (calf 1)

14->16 (calf 2)
"""


def load_frame(src_file, myCamera, measurements):

    array = np.load(src_file)
    pose = BDPose.PoseSet(array)

    pose.add_measurements(measurements)
    direction = -1
    depth = 0
    # hip_center = myCamera.calculate_xyz(pose.hip_center, depth)
    hip_center = [pose.hip_center[0], pose.hip_center[1],depth]

    img = np.asarray(Image.open(src_file.split(".npy")[0] + ".jpg"))
    plt.imshow(img)

    #left leg
    hip_left = apply_measurement(myCamera, pose, pose.pts[11], hip_center, pose.hip_width/2, direction)
    knee_left = apply_measurement(myCamera, pose, pose.pts[13], hip_left, pose.thigh_height, direction)
    ankle_left = apply_measurement(myCamera, pose, pose.pts[15], knee_left, pose.calf_height, direction)

    left_leg = [hip_center, hip_left, knee_left, ankle_left]

    #right leg
    hip_right = apply_measurement(myCamera, pose, pose.pts[12], hip_center, pose.hip_width/2, -direction)
    knee_right = apply_measurement(myCamera, pose, pose.pts[14], hip_right, pose.thigh_height, -direction)
    ankle_right = apply_measurement(myCamera, pose, pose.pts[16], knee_right, pose.calf_height, -direction)

    right_leg = [hip_center, hip_right, knee_right, ankle_right]


    #left side
    shoulder_left = apply_measurement(myCamera, pose, pose.pts[5], hip_left, pose.torso_height, direction)


    #right side
    shoulder_right = apply_measurement(myCamera, pose, pose.pts[6], hip_right, pose.torso_height, direction)


    return [ankle_right, knee_right, hip_right, hip_center, hip_left, knee_left, ankle_left]


def main(source):

    f = open("legs.csv", 'w')

    filenames = []
    order = []
    excluded = []

    thigh = []
    shoulders = []
    calf = []
    hip = []


    for filename in os.listdir(source):
        if ".npy" in filename:
            filenames.append(filename)
            order.append(float(filename.split("frame")[-1].split(".npy")[0]))
            array = np.load(source + "/" + filename)
            pose = BDPose.PoseSet(array)

            thigh.append(np.linalg.norm(pose.thigh01))
            thigh.append(np.linalg.norm(pose.thigh02))
            shoulders.append(np.linalg.norm(pose.shoulders))
            calf.append(np.linalg.norm(pose.calf01))
            calf.append(np.linalg.norm(pose.calf02))
            hip.append(np.linalg.norm(pose.hip))

    print(f"Thigh max dimension {np.max(thigh)}")
    print(f"Shoulders max dimension {np.max(shoulders)}")
    print(f"Calf max dimension {np.max(calf)}")
    print(f"Hip max dimension {np.max(hip)}")

    # measurements  = {
    #     "shoulder width": np.max(shoulders),
    #     "hip width": np.max(hip),
    #     "forearm height": np.max(calf) * 12/17,
    #     "upperarm height": np.max(calf) * 12/17,
    #     "torso height": np.max(thigh) * 20/19,
    #     "thigh height": np.max(thigh),
    #     "calf height": np.max(calf)
    # }

    dim = 500

    measurements  = {
        "shoulder width": dim * 8/19,
        "hip width": dim * 10/19,
        "forearm height": dim * 12/19,
        "upperarm height": dim * 12/19,
        "torso height": dim * 20/19,
        "thigh height": dim,
        "calf height": dim * 17/19
    }

    filenames = np.array(filenames)
    filenames = filenames[np.argsort(np.array(order))]

    myCamera = Camera.PinholeCamera(1920, 1080, 10000, 10000, 0 , 0)

    for filename in filenames:
        if ".npy" in filename:
            try:
                legs = load_frame(source + "/" + filename, myCamera, measurements)
                for i in range(len(legs)):
                    x = str(legs[i][0])
                    y = str(legs[i][1])
                    z = str(legs[i][2])
                    f.write(x)
                    f.write(" ")
                    f.write(y)
                    f.write(" ")
                    f.write(z)
                    if i < len(legs) - 1:
                        f.write(",")
                    else:
                        f.write("\n")
            except:
                excluded.append(filename)

    print(len(excluded))


main("out_image")
