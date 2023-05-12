
import argparse
import os
import json
import cv2
import numpy as np
from annotation_stat import open_spire_annotations


def main():
    parser = argparse.ArgumentParser(description="Statistic on spire annotations")
    parser.add_argument(
        "--spire-anno",
        default="/home/jario/dataset/BB200326_mbzirc_c1_yellow_train_BDT/annotations",
        help="path to spire annotation dir",
        # required=True
    )
    parser.add_argument(
        "--area-min",
        default=0,
        help="retain the boxes that area >= 'area-min'",
        # required=True
    )
    parser.add_argument(
        "--area-max",
        default=1e8,
        help="retain the boxes that area <= 'area-max'",
        # required=True
    )
    args = parser.parse_args()
    image_jsons = open_spire_annotations(args.spire_anno)

    if os.path.exists(os.path.join(args.spire_anno, 'annotations')):
        args.spire_anno = os.path.join(args.spire_anno, 'annotations')

    min_area = 1e6
    for image_anno in image_jsons:
        retained_annos = []
        if len(image_anno['annos']) == 0:
            print("[WARN] {} has no objects".format(image_anno['file_name']))

        for anno in image_anno['annos']:
            bbox = np.array(anno['bbox'], np.float32)
            retain = True
            if bbox[2] == 0 or bbox[3] == 0:
                print("File_name: {}, bbox: {}".format(image_anno['file_name'], bbox))
                retain = False

            area = bbox[2] * bbox[3]
            min_area = min(area, min_area)
            # print(min_area)

            x, y, w, h = anno['bbox']
            if x < 0 or y < 0:
                print("----x: {}, y: {}".format(x, y))
            x = max(0, x)
            y = max(0, y)
            anno['bbox'] = [x, y, w, h]

            if float(area) < float(args.area_min) or float(area) > float(args.area_max):
                print("File_name: {}, area: {}".format(image_anno['file_name'], area))
                retain = False
            if retain:
                retained_annos.append(anno)
        image_anno['annos'] = retained_annos

        fp = open(os.path.join(args.spire_anno, image_anno['file_name']+'.json'), 'w')
        json.dump(image_anno, fp)

    print("min_area: {}".format(min_area))


if __name__ == '__main__':
    main()
