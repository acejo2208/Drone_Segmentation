import os
import torch.nn as nn 
import cv2 
import json 
import pandas as pd 
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIAlign, RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

import pycocotools.mask as maskUtils
from skimage import measure
 


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # Use torchvision ops for CPU mode instead
        for m in model.modules():
            if isinstance(m, (RoIPool, RoIAlign)):
                if not m.aligned:
                    # aligned=False is not implemented on CPU
                    # set use_torchvision on-the-fly
                    m.use_torchvision = True
        warnings.warn('We set use_torchvision=True in CPU mode.')
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    # prepare data
    data = dict(img_info=dict(filename=img), img_prefix=None)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()


def json_generation(file_name, img, result):
    with open('Result_json/'+ os.path.splitext(file_name)[0]+'.json', 'w') as fp:
        fp.write("{")
        fp.write('"shape"')
        fp.write(':')
        fp.write('[\n')
    
    CLASSES = ('lawn, flower_garden', 'forest', 'river', 'road', 'pavement',
               'parking_lot', 'crosswalk', 'hiking_trail', 'trail', 'flower_bed')

    segm_json_results = []

    det, seg = result[0], result[1]
    for label in range(len(det)):
        bboxes = det[label]

        if isinstance(seg, tuple):
            segms = seg[0][label]
            mask_score = seg[1][label]
        else:
            segms = seg[label]
            mask_score = [bbox[4] for bbox in bboxes]
        for i in range(bboxes.shape[0]):
            data = dict()

            data['label'] = CLASSES[label]

            data['points'] = binary_to_polygon(segms[i])
            data['shape_type'] = "polygon"

            segm_json_results.append(data)
            # writing JSON object
            with open('Result_json/'+os.path.splitext(file_name)[0]+'.json', 'a') as f:
                pd.Series(data).to_json(f, indent=2)

                f.write(',')

    # writing JSON object
    with open('Result_json/'+os.path.splitext(file_name)[0]+'.json', 'rb+') as g:
        g.seek(-1, os.SEEK_END)
        g.truncate()

    # writing JSON object
    with open('Result_json/'+os.path.splitext(file_name)[0]+'.json', 'a') as g:
        g.write('],\n')
        g.write('"imagePath":')
        g.write('"')
        g.write(file_name)
        g.write('"')
        g.write(',\n')
        g.write('"imageData": null,\n')
        g.write('"imageHeight": 2160,\n')
        g.write('"imageWidth": 3840\n')
        g.write('}')

def binary_to_polygon(binary_mask):
    segmentation = []
    segmentation_polygons = []
    contours = measure.find_contours(binary_mask, 0.5)
    #print('size of contours', contours)
    for contour in contours:
        contour = contour[::200]
        contour = np.flip(contour, axis=1)
        #print('ravel ?', contour.ravel())
        contour = contour.ravel().tolist()
        segmentation.append(contour)

    for i, segment in enumerate(segmentation):
        poligon = []
        poligons = []
        for j in range(len(segment)):
            poligon.append(segment[j])
            if (j+1) % 2 == 0:
                poligons.append(poligon)
                poligon = []
        segmentation_polygons.append(poligons)

    return segmentation_polygons

def cv2_polygons(binary_mask):
    segmentation = []
    segmentation_polygons = []
    contours, hierarchy = cv2.findContours(
        (binary_mask).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)


    # Get the contours
    for contour in contours:
        #contour = contour[:10]
        contour = contour.flatten().tolist()
        segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)

    # Get the polygons as (x, y) coordinates
    for i, segment in enumerate(segmentation):
        poligon = []
        poligons = []
        for j in range(len(segment)):
            poligon.append(segment[j])
            if (j+1) % 2 == 0:
                poligons.append(poligon)
                poligon = []
            #print(poligon)
        segmentation_polygons.append(poligons)

    return segmentation_polygons

def pycococreator(data, binary_mask):

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(
       binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance=2.5)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    data['points'] = polygons

    return data['points']

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def coco_style(binary_mask):
    binary_mask = np.asfortranarray(binary_mask)
    mask_rle = maskUtils.encode(binary_mask)
    points = mask_rle['counts'].decode()
    return points


def json_generation_float(file_name, img, result):
    with open('single_json/' + os.path.splitext(file_name)[0]+'.json', 'w') as fp:
        fp.write("{")
        fp.write('"shapes"')
        fp.write(':')
        fp.write('[\n')

    CLASSES = ('lawn, flower_garden', 'forest', 'river', 'road', 'pavement',
               'parking_lot', 'crosswalk', 'hiking_trail', 'trail', 'flower_bed')

    #bbox_json_results = []
    segm_json_results = []

    det, seg = result[0], result[1]
    #print('det size', len(det))
    for label in range(len(det)):
        bboxes = det[label]
        #print(bboxes.shape)
        if isinstance(seg, tuple):
            segms = seg[0][label]
            mask_score = seg[1][label]
        else:
            segms = seg[label]
            mask_score = [bbox[4] for bbox in bboxes]
        #print('bb size', bboxes.shape[0])
        for i in range(bboxes.shape[0]):
            data = dict()

            data['label'] = CLASSES[label]
            #data['score'] = float(mask_score[i])

            multi_array = binary_to_polygon(segms[i])
            #print(len(points))

            points = []
            for i in range(len(multi_array)):
                for j in multi_array[i]:
                    points.append(j)

            data['points'] = points
            
            data['shape_type'] = "polygon"

            segm_json_results.append(data)
            # writing JSON object
            with open('single_json/'+os.path.splitext(file_name)[0]+'.json', 'a') as f:
                pd.Series(data).to_json(f, indent=1)
                #json.dump(data, f, indent=4)
                #json.dump(dumped, f)

                f.write(',')

    # writing JSON object
    with open('single_json/'+os.path.splitext(file_name)[0]+'.json', 'rb+') as g:
        g.seek(-1, os.SEEK_END)
        g.truncate()

    # writing JSON object
    with open('single_json/'+os.path.splitext(file_name)[0]+'.json', 'a') as g:
        g.write('],\n')
        g.write('"imagePath":')
        g.write('"')
        g.write(file_name)
        g.write('"')
        g.write(',\n')
        g.write('"imageData": null,\n')
        g.write('"imageHeight": 2160,\n')
        g.write('"imageWidth": 3840\n')
        g.write('}')
