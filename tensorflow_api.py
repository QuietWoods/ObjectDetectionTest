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
import numpy as np
import time
import tensorflow as tf
from PIL import Image
import cv2
import os

from model_metric import load_index_class
from draw_bbs import DrawBoundingBoxes


import logging

logger = logging.getLogger("main")
# logger.setLevel(logging.DEBUG)  root logger leval is higher than info
# logging.StreamHandler(sys.stdout)
# logger.setFileLevel(logging.INFO)

brand2package = {
    "1": "1",
    "2": "1",
    "3": "1",
    "4": "2",
    "5": "3",
    "6": "2",
    "7": "2",
    "8": "2",
    "9": "3",
    "10": "3",
    "11": "1",
    "12": "1",
    "13": "1",
    "14": "1",
    "15": "1"
}


# # Detection
class ObjectDetection(object):
    def __init__(self, model_path, label_path, class2index_path, num_classes, score_threshold, output, shape_flag, save_image_size=None):
        self.save_image_size = save_image_size
        # ## Load a (frozen) Tensorflow model into memory.
        self.image_tensor = None
        self.tensor_dict = {}

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)
        self.__create_image_tensor()

        self.score_threshold = score_threshold
        self.predict_result_file = os.path.join(os.path.dirname(output), "predict_data_details.csv")
        self.result_w = open(self.predict_result_file, "w")
        self.result_w.write("filename,width,height,class,xmin,ymin,xmax,ymax,score,index\n")
        self.index2class = load_index_class(class2index_path)
        self.draw_obj = DrawBoundingBoxes(label_path=label_path, num_classes=num_classes)
        self.shape_flag = shape_flag

    def close(self):
        self.sess.close()
        self.result_w.close()

    def __write(self, output_dict):
        boxes = output_dict['detection_boxes']
        im_width = output_dict['width']
        im_height = output_dict['height']
        for i in range(output_dict['num_detections']):
            if output_dict['detection_scores'][i] >= self.score_threshold:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                xmin, xmax, ymin, ymax = (int(xmin * im_width), int(xmax * im_width),
                                        int(ymin * im_height), int(ymax * im_height))
                self.result_w.write("{},{},{},{},{},{},{},{},{:.4f},{}\n".format(
                    output_dict['filename'],
                    output_dict['width'],
                    output_dict['height'],
                    self.index2class[str(output_dict['detection_classes'][i])],
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    output_dict['detection_scores'][i],
                    i))

    def __create_image_tensor(self):
        # Get handles to input and output tensors
        with self.detection_graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def run_inference_for_single_image(self, image):
        # Get handles to input and output tensors
        # Run inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def detect(self, image_path, output):
        filename = os.path.basename(image_path)
        result_path = os.path.join(output, filename)

        per_image_start = time.time()
        # Load image
        # keep a consistent pre_processing and inference
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.draw_obj.load_image_into_numpy_array_fixed(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        # object detect
        detect_ts = time.time()
        output_dict = self.run_inference_for_single_image(image_np)
        detect_time = time.time() - detect_ts
        logger.info("inference time: ", detect_time)

        draw_detect_classes = output_dict['detection_classes']
        if self.shape_flag:
            draw_detect_classes = [int(brand2package[str(i)]) for i in draw_detect_classes]

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
