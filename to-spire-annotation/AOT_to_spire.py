
import argparse
import os
import json
import cv2
from tqdm import tqdm
import pycocotools.mask as maskUtils
import numpy as np
import cv2
import math


def find_contours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy


def solve_coco_segs(segs_anno, h, w):
    assert type(segs_anno['counts']) == list, "segs_anno['counts'] should be list"
    rle = maskUtils.frPyObjects(segs_anno, h, w)
    mask = maskUtils.decode(rle)
    contours, hierarchy = find_contours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contour in contours:
        area += int(round(cv2.contourArea(contour)))
    if area != 0:
        segmentations = []
        for contour in contours:
            if contour.shape[0] >= 6:  # three points
                segmentation = []
                for cp in range(contour.shape[0]):
                    segmentation.append(int(contour[cp, 0, 0]))
                    segmentation.append(int(contour[cp, 0, 1]))
                segmentations.append(segmentation)
    else:
        segmentations = []
    return segmentations


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotation to spire annotation")
    parser.add_argument(
        "--coco-anno",
        default="E:/airborne-detection-starter-kit/utility/data/part1/ImageSets/groundtruth.json",
        help="path to coco annotation file",
        # required=True
    )
    parser.add_argument(
        "--coco-image-dir",
        default='E:/airborne-detection-starter-kit/utility/data/part1/Images',
        help="path to coco image dir",
    )
    parser.add_argument(
        "--output-dir",
        default="F:/BB-AirborneObjectTracking-v221109",
        help="path to spire home dir",
    )
    parser.add_argument(
        '--save-image',
        action='store_true',
        help='save image to scaled_images'
    )
    parser.add_argument(
        '--show-image',
        action='store_true',
        help='show image for testing'
    )
    args = parser.parse_args()

    image_root = args.coco_image_dir
    f = open(args.coco_anno, 'r')
    json_str = f.read()
    json_dict = json.loads(json_str)

    categories = []
    sub_dirs = []
    entity_all = 0
    entity_has_id = 0
    for i, flight_id in enumerate(json_dict['samples'].items()):
        metadata = flight_id[1]['metadata']
        data_path = metadata['data_path']
        sub_dir = data_path.split('/')[0]
        if sub_dir not in sub_dirs:
            sub_dirs.append(sub_dir)
        flight_h = metadata['resolution']['height']
        flight_w = metadata['resolution']['width']
        entities = flight_id[1]['entities']

        frame_ids = dict()
        frame_entities = dict()
        for j, entity in enumerate(entities):
            assert entity['flight_id'] == flight_id[0]
            if 'id' in entity.keys():
                assert 'time' in entity.keys()
                assert 'bb' in entity.keys()
                assert 'labels' in entity.keys()
                assert 'blob' in entity.keys()
                assert 'img_name' in entity.keys()
                assert entity['labels']['is_above_horizon'] in [-1, 0, 1]

                img_fn = os.path.join(image_root, entity['flight_id'], entity['img_name'])
                if not os.path.exists(img_fn):
                    print(entity['flight_id'], entity['img_name'])

                range_distance_m = 0
                if 'range_distance_m' in entity['blob'].keys():
                    range_distance_m = entity['blob']['range_distance_m']
                if math.isnan(range_distance_m):
                    range_distance_m = -1
                entity['range_distance_m'] = range_distance_m

                frame_id = entity['blob']['frame']
                if frame_id in frame_ids.keys():
                    frame_ids[frame_id].append(entity['id'])
                    frame_entities[frame_id].append(entity)
                    # print('{} -> {}'.format(flight_id[0], frame_id))
                else:
                    frame_ids[frame_id] = [entity['id']]
                    frame_entities[frame_id] = [entity]

                _id = entity['id']
                if _id not in categories:
                    categories.append(_id)
                entity_has_id += 1
            entity_all += 1

        if len(frame_ids) > 0:
            video_dir = flight_id[0]
            scaled_images = os.path.join(args.output_dir, video_dir, 'scaled_images')
            annotations = os.path.join(args.output_dir, video_dir, 'annotations')
            if not os.path.exists(annotations):
                os.makedirs(annotations)
            if not os.path.exists(scaled_images):
                os.makedirs(scaled_images)

            category2tid = dict()
            for k, cat in enumerate(categories):
                category2tid[cat] = k + 1
            for k, (entity_id_list, entity_list) in enumerate(zip(frame_ids.values(), frame_entities.values())):
                entity_0 = entity_list[0]
                img_fn = os.path.join(image_root, entity_0['flight_id'], entity_0['img_name'])
                img = cv2.imread(img_fn)

                spire_dict = dict()
                # entity_0['img_name']
                spire_dict['file_name'] = "{}_{}".format(video_dir, str(entity_0['blob']['frame']).zfill(8)) + \
                                          os.path.splitext(entity_0['img_name'])[1]
                spire_dict['height'], spire_dict['width'] = img.shape[0], img.shape[1]
                spire_dict['frame'] = entity_0['blob']['frame']
                spire_dict['time'] = entity_0['time']
                spire_dict['annos'] = []

                for entity_id, entity in zip(entity_id_list, entity_list):
                    spire_anno = dict()
                    spire_anno['area'] = entity['bb'][2] * entity['bb'][3]
                    spire_anno['bbox'] = entity['bb']
                    category_name = 'None'
                    if entity['id'].startswith('Airplane'):
                        category_name = 'Airplane'
                    elif entity['id'].startswith('Helicopter'):
                        category_name = 'Helicopter'
                    elif entity['id'].startswith('Bird'):
                        category_name = 'Bird'
                    elif entity['id'].startswith('Airborne'):
                        category_name = 'Airborne'
                    elif entity['id'].startswith('Drone'):
                        category_name = 'Drone'
                    elif entity['id'].startswith('Flock'):
                        category_name = 'Flock'
                    else:
                        assert False, entity['id']

                    spire_anno['category_name'] = category_name
                    spire_anno['tracked_id'] = category2tid[entity['id']]
                    spire_anno['is_above_horizon'] = entity['labels']['is_above_horizon']
                    spire_anno['range_distance_m'] = entity['range_distance_m']
                    spire_dict['annos'].append(spire_anno)

                output_fn = os.path.join(annotations, spire_dict['file_name'] + '.json')
                with open(output_fn, "w") as f:
                    json.dump(spire_dict, f)
                open(os.path.join(scaled_images, spire_dict['file_name']), 'wb').write(open(img_fn, 'rb').read())


if __name__ == '__main__':
    main()
