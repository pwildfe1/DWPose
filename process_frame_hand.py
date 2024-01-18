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

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

register_all_modules()




def process_frame(model, visualizer, frame, out_file):

    # det_result = inference_detector(detector, frame)
    # pred_instance = det_result.pred_instances.cpu().numpy()
    # bboxes = np.concatenate(
    #     (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    # bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
    #                                pred_instance.scores > args.bbox_thr)]
    # bboxes = bboxes[nms(bboxes, args.nms_thr), :4]


    # inference a single image
    batch_results = inference_topdown(model, frame)
    print(batch_results)
    results = merge_data_samples(batch_results)

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
    # model.cfg.visualizer.radius = 8
    # model.cfg.visualizer.alpha = .8
    # model.cfg.visualizer.line_width = 3

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # visualizer.set_dataset_meta(
    #     model.dataset_meta, skeleton_style="mmpose")

    return [model, visualizer]

"""
11->12 (hips)

11->13 (thigh 1)

12->14 (thigh 2)

13->15 (calf 1)

14->16 (calf 2)
"""


def main():
    config_file = 'rtmdet_nano_320-8xb32_hand.py'
    checkpoint_file = 'hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth'

    det_config = "rtmpose-m_8xb256-210e_hand5-256x256.py"
    det_checkpoint = "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"

    # build detector
    # detector = init_detector(det_config, det_checkpoint)
    # detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    model, visualizer = build_model(config_file, checkpoint_file)

    frame = "single_frame.png"
    out_file = "overlay_frame.png"

    results = process_frame(model, visualizer, frame, out_file)
    # out_file = "out_image/" + os.path.split(frame)[1]
    img = plt.imread(out_file)

    points = results.pred_instances.keypoints[0]


main()