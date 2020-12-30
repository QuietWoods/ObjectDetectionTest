#!/usr/bin python  
# -*- coding:utf-8 -*-  
""" 
@author: quietwoods 
@file: object_detection.py 
@time: 2020/12/25
@contact: wanglei2xf@163.com
@site:  
@software: PyCharm 
"""
import time
import os
import sys
from tensorflow_api import ObjectDetection
from tensorflowlite_api import TensorflowLiteAPI

import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)


def detect_pipeline(images,
                    model_path, label_path,
                    class2index_path,
                    num_classes, score_threshold,
                    output, quantized,
                    save_image_size, shape_flag=False,
                    test_num=10):
    ts = time.time()
    if not os.path.exists(output):
        os.makedirs(output)

    test_image_paths = [os.path.join(images, image) for image in os.listdir(images)]

    if quantized:
        detect_obj = TensorflowLiteAPI(model_path, label_path, class2index_path, num_classes, score_threshold, output, save_image_size, tf_ob_api_is_quantized=True)
    else:
        detect_obj = ObjectDetection(model_path, label_path, class2index_path, num_classes, score_threshold, output, shape_flag, save_image_size)

    test_count = 0
    total_test_num = len(test_image_paths)
    for image_path in test_image_paths:
        if test_count < test_num:
            detect_obj.detect(image_path, output)
        else:
            break
        test_count += 1
        logger.info("test {}/{}.".format(test_count, total_test_num))

    total_time = time.time() - ts
    logger.error('Testing {} pictures cost {:.4f}s.'.format(test_count, total_time))

    detect_obj.close()

    return True

