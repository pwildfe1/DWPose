import os
import logging
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.patches import Circle

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

register_all_modules()




def process_frame(model, visualizer, frame, out_file):

    # inference a single image
    batch_results = inference_topdown(model, frame)
    results = merge_data_samples(batch_results)

    # print(results.pred_instances.keypoints)

    # show the results
    img = imread(frame, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        show=False,
        show_kpt_idx=False,
        out_file=out_file)

    return results



def build_model(config_file, checkpoint_file):

    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    # init visualizer
    model.cfg.visualizer.radius = 8
    model.cfg.visualizer.alpha = .8
    model.cfg.visualizer.line_width = 3

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style="mmpose")

    return [model, visualizer]

"""
11->12 (hips)

11->13 (thigh 1)

12->14 (thigh 2)

13->15 (calf 1)

14->16 (calf 2)
"""


def main(sequence_name):
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    model, visualizer = build_model(config_file, checkpoint_file)

    # self.coco_joints_name = (
    #         'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
    #         'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')

    files = []
    indices = []

    source_directory = f"./frames/{sequence_name}/"
    out_directory = f"./out_image/{sequence_name}/"

    for frame in os.listdir(source_directory):
        indices.append(int(os.path.splitext(frame)[0].split("_")[-1]))
        files.append(frame)
    
    indices = np.array(indices)
    files = list(np.array(files)[np.argsort(indices)])

    for frame in tqdm(files, desc="Processing files"):
        if os.path.exists(out_directory) == False:
            os.mkdir(out_directory)
        out_file = out_directory + os.path.split(frame)[1]
        results = process_frame(model, visualizer, source_directory + frame, out_file)
        img = plt.imread(out_file)

        points = results.pred_instances.keypoints[0]

        # print(points.shape)

        joints = np.zeros((19,2))

        joints[0:17] = points[0:17]
        joints[17] = (points[11] + points[12])/2
        joints[18] = (points[5] + points[6])/2

        
        # circ = plt.Circle((points[p0,0], points[p0,1]),5)
        # circ.set_color([0, 0, 1])
        # ax.add_patch(circ)

        # circ = plt.Circle((joints[15,0], joints[15,1]),5)
        # circ.set_color([0, 0, 1])
        # ax.add_patch(circ)
        # plt.show()

        np.save(os.path.splitext(out_file)[0] + ".npy", joints)

        # break


main("SixStep")