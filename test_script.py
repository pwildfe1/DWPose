from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

register_all_modules()



def main(img_file):
    draw_out = True

    # build the model from a config file and a checkpoint file
    if draw_out:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    # init visualizer
    model.cfg.visualizer.radius = 3
    model.cfg.visualizer.alpha = .8
    model.cfg.visualizer.line_width = 1

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style="mmpose")

    # inference a single image
    batch_results = inference_topdown(model, img_file)
    results = merge_data_samples(batch_results)

    out_file = img_file.split(".")[0] + "_out.jpg"

    # show the results
    img = imread(img_file, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        show=True,
        show_kpt_idx=False,
        out_file=out_file)

    if out_file is not None:
        print_log(
            f'the output image has been saved at {out_file}',
            logger='current',
            level=logging.INFO)


main("demo.jpg")