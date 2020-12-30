#!/usr/bin python  
# -*- coding:utf-8 -*-  
""" 
@author: quietwoods 
@file: draw_boundingboxes.py 
@time: 2020/12/28
@contact: wanglei2xf@163.com
@site:  
@software: PyCharm 
"""
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import logging

logger = logging.getLogger("main")
# logger.setLevel(logging.DEBUG)  root logger leval is higher than info
# logger.info('info')
# logger.warning('war')
# logger.error('err')
# logging.StreamHandler(sys.stdout)
# logger.setFileLevel(logging.INFO)


class DrawBoundingBoxes(object):
    def __init__(self, label_path, num_classes):
        label_map = label_map_util.load_labelmap(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    @classmethod
    def load_image_into_numpy_array_fixed(cls, image):
        (im_width, im_height) = image.size
        return np.asarray(image).reshape((im_height, im_width, 3)).astype(np.uint8)

    def draw(self, image_np, output_dict, draw_detect_classes, score_threshold):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            draw_detect_classes,
            output_dict['detection_scores'],
            self.category_index,
            min_score_thresh=score_threshold,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
