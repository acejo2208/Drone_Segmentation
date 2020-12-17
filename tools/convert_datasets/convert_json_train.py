import argparse
import glob
import os.path as osp

#import cityscapesscripts.helpers.labels as CSLabels
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from collections import namedtuple

Label = namedtuple('Label', ['name', 'id', 'trainId', 'category',
    'categoryId', 'hasInstances', 'ignoreInEval', 'color',])

labels = [
    #       name                     id    trainId   category            
    #     catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            
        , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'background'           , 1  ,       255, 'void'             
        , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'lawn, flower_garden'  , 2  ,        0 , 'nature'            
        , 1       , True        , False        , (213,121,219) ),
    Label(  'forest'               , 3  ,        1 , 'nature'            
        , 1       , True        , False        , (4, 200, 3) ),
    Label(  'liver'                , 4  ,        2 , 'nature'            
        , 1       , True        , False        , (232,250,187) ),
    Label(  'road'                 , 5  ,        3 , 'flat'            
        , 2       , True        , False        , (128, 64,128) ),
    Label(  'pavement'             , 6  ,        4 , 'flat'            
        , 2       , True        , False        , (244, 35,232) ),
    Label(  'parking_lot'          , 7  ,        5 , 'flat'            
        , 2       , True        , False        , (250,170,160) ),
    Label(  'crosswalk'            , 8  ,        6 , 'flat'            
        , 2       , True        , False        , (252,239,44) ),
    Label(  'hiking_trail'         , 9  ,        7 , 'flat'            
        , 2       , True        , False        , (251,181,244) ),
    Label(  'trail'                , 10  ,        8 , 'flat'            
        , 2       , True        , False        , (245,162,214) ),
    Label(  'flower_bed'           , 11  ,       9 , 'nature'           
        ,1       , True        , False        , (201,243, 98) )  
]

name2label      = { label.name    : label for label in labels}
# id to label object
id2label        = { label.id      : label for label in labels}
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels)}


category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]



def collect_files(img_dir, gt_dir):
    suffix = '.jpg'
    files = []
    for img_file in glob.glob(osp.join(img_dir, '*.jpg')):
        assert img_file.endswith(suffix), img_file
        inst_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + '_gtFine_instanceIds.png'
        # Note that labelIds are not converted to trainId for seg map
        segm_file = gt_dir + img_file[
            len(img_dir):-len(suffix)] + '_gtFine_labelIds.png'
        files.append((img_file, inst_file, segm_file))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    print('Loading annotation images')
    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    img_file, inst_file, segm_file = files
    inst_img = mmcv.imread(inst_file, 'unchanged')
    # ids < 24 are stuff labels (filtering them first is about 5% faster)
    unique_inst_ids = np.unique(inst_img[inst_img >= 24])
    anno_info = []
    for inst_id in unique_inst_ids:
        # For non-crowd annotations, inst_id // 1000 is the label_id
        # Crowd annotations have <1000 instance ids
        label_id = inst_id // 1000 if inst_id >= 1000 else inst_id
        label = id2label[label_id]
        if not label.hasInstances or label.ignoreInEval:
            continue

        category_id = label.id
        iscrowd = int(inst_id < 1000)
        mask = np.asarray(inst_img == inst_id, dtype=np.uint8, order='F')
        mask_rle = maskUtils.encode(mask[:, :, None])[0]

        area = maskUtils.area(mask_rle)
        # convert to COCO style XYWH format
        bbox = maskUtils.toBbox(mask_rle)

        # for json encoding
        mask_rle['counts'] = mask_rle['counts'].decode()

        anno = dict(
            iscrowd=iscrowd,
            category_id=category_id,
            bbox=bbox.tolist(),
            area=area.tolist(),
            segmentation=mask_rle)
        anno_info.append(anno)
    #video_name = osp.basename(osp.dirname(img_file))
    img_info = dict(
        # remove img_prefix for filename
        file_name=osp.basename(img_file),
        height=inst_img.shape[0],
        width=inst_img.shape[1],
        anno_info=anno_info,
        segm_file=osp.basename(segm_file))

    return img_info


def cvt_annotations(image_infos, out_json_name):
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['categories'] = []
    out_json['annotations'] = []
    for image_info in image_infos:
        image_info['id'] = img_id
        anno_infos = image_info.pop('anno_info')
        out_json['images'].append(image_info)
        for anno_info in anno_infos:
            anno_info['image_id'] = img_id
            anno_info['id'] = ann_id
            out_json['annotations'].append(anno_info)
            ann_id += 1
        img_id += 1
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            cat = dict(id=label.id, name=label.name)
            out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Drone annotations to COCO format')
    parser.add_argument('cityscapes_path', help='drone data path')
    parser.add_argument('--img-dir', default='leftImg8bit', type=str)
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(cityscapes_path, args.img_dir)
    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    set_name = dict(
        train='instancesonly_filtered_gtFine_train.json',
        val='instancesonly_filtered_gtFine_val.json',
        test='instancesonly_filtered_gtFine_test.json')

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(
                print_tmpl='It tooks {}s to convert Cityscapes annotation'):
            files = collect_files(
                osp.join(img_dir, split), osp.join(gt_dir, split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            cvt_annotations(image_infos, osp.join(out_dir, json_name))


if __name__ == '__main__':
    main()
