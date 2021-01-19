#!/usr/bin python  
# -*- coding:utf-8 -*-  
""" 
@author: quietwoods 
@file: tensorflowlite_api.py 
@time: 2020/12/28
@contact: wanglei2xf@163.com
@site:  
@software: PyCharm 
"""
import time
import tensorflow as tf
import numpy as np
import cv2
import os

from PIL import Image

from model_metric import load_index_class
from draw_bbs import DrawBoundingBoxes

import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print(tf.__version__)


class TensorflowLiteAPI(object):
    def __init__(self,
                 model_path,
                 label_path,
                 class2index_path,
                 num_classes,
                 score_threshold,
                 output,
                 save_image_size=None,
                 tf_ob_api_is_quantized=False):
        self.input_mean = 127.5
        self.input_std = 127.5

        self.model_path = model_path
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.contrib.lite.Interpreter(model_path=model_path)  # converted_model.tflite
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        # print(self.input_details)
        input_dtype = self.input_details[0]['dtype']

        if tf_ob_api_is_quantized:
            assert input_dtype == np.uint8, "model is not quantized uint8!"
        else:
            assert input_dtype == np.float32, "model is not quantized float32!"

        self.tf_ob_api_is_quantized = tf_ob_api_is_quantized
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

        self.score_threshold = score_threshold
        self.predict_result_file = os.path.join(os.path.dirname(output), "predict_data_details.csv")
        self.result_w = open(self.predict_result_file, "w")
        self.result_w.write("filename,width,height,class,xmin,ymin,xmax,ymax,score,index\n")
        self.index2class = load_index_class(class2index_path)
        self.draw_obj = DrawBoundingBoxes(label_path=label_path, num_classes=num_classes)
        self.save_image_size = save_image_size

    def close(self):
        self.result_w.close()

    def __write(self, output_dict):
        im_width, im_height = output_dict['width'], output_dict['height']
        boxes = output_dict['detection_boxes']
        for i in range(output_dict['num_detections']):
            if output_dict['detection_scores'][i] >= self.score_threshold:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                xmin, xmax, ymin, ymax = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
                self.result_w.write("{},{},{},{},{},{},{},{},{:.4f},{}\n".format(
                    output_dict['filename'],
                    im_width,
                    im_height,
                    self.index2class[str(output_dict['detection_classes'][i] + 1)],
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    output_dict['detection_scores'][i],
                    i))

    def run_tensorflowlite_inference_for_single_image(self, image):
        cropped_image = image.resize((self.width, self.height))
        # add N dim
        input_data = np.expand_dims(cropped_image, axis=0)
        # Test the model on random input data.
        if self.tf_ob_api_is_quantized:
            # input_data = np.array(np.random.random_sample(self.input_shape), dtype=np.uint8)
            input_data = np.uint8(input_data)
        else:
            # input_data = np.array(np.random.random_sample(self.input_shape), dtype=np.float32)
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_dict = {'detection_boxes': np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index'])),
                       'detection_scores': np.squeeze(self.interpreter.get_tensor(self.output_details[2]['index'])),
                       'detection_classes': np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index'])).astype(np.uint8),
                       'num_detections': int(np.squeeze(self.interpreter.get_tensor(self.output_details[3]['index'])))}

        # all outputs are float32 numpy arrays, so convert types as appropriate
        # deal with abnormal detection scores
        src_detection_scores = output_dict['detection_scores']
        # print(self.output_details)
        # print(output_dict['detection_boxes'])
        # print(src_detection_scores)
        # test_ts = time.time()
        output_dict['detection_scores'] = np.where(src_detection_scores > 1, 0, src_detection_scores)
        # print(output_dict['detection_scores'])
        # print('clip time: ', time.time() - test_ts)

        # print(output_dict['detection_classes'])
        # print(output_dict['num_detections'])

        # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        return output_dict

    def detect(self, image_path, output):
        print(image_path)
        filename = os.path.basename(image_path)
        result_path = os.path.join(output, filename)

        per_image_start = time.time()
        # Load image
        # keep a consistent pre_processing and inference
        # image = Image.open(image_path).resize((self.width, self.height))
        image = Image.open(image_path)
        image_np = self.draw_obj.load_image_into_numpy_array_fixed(image)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        # object detect
        detect_ts = time.time()
        output_dict = self.run_tensorflowlite_inference_for_single_image(image)
        detect_time = time.time() - detect_ts
        logger.info("inference time: ", detect_time)

        draw_detect_classes = output_dict['detection_classes'] + 1

        # Visualization of the results of a detection.
        self.draw_obj.draw(image_np, output_dict, draw_detect_classes, self.score_threshold)

        if self.save_image_size:
            resized = cv2.resize(image_np, self.save_image_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image_np
        cv2.imwrite(result_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

        per_image_end = time.time()
        logger.info('detect per picture time: ', per_image_end - per_image_start)

        output_dict['filename'] = filename
        output_dict['height'], output_dict['width'], _ = image_np.shape

        self.__write(output_dict)

