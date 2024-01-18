import os
import logging
import json
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




def process_frame(model, visualizer, frame):

    # inference a single image
    batch_results = inference_topdown(model, frame)
    results = merge_data_samples(batch_results)

    # print(results.pred_instances.keypoints)

    out_file = "out_image/" + os.path.split(frame)[1]

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

    if out_file is not None:
        print_log(
            f'the output image has been saved at {out_file}',
            logger='current',
            level=logging.INFO)

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


def main():
    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    model, visualizer = build_model(config_file, checkpoint_file)

    # self.coco_joints_name = (
    #         'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
    #         'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')

    for frame in os.listdir("source_image"):
        results = process_frame(model, visualizer, "source_image/" + frame)
        out_file = "out_image/" + os.path.split(frame)[1]
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


main()

t = np.load("out_image/frame0.npy")
print(t.shape)