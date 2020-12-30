# coding: utf-8
import numpy as np
import time
import os
import argparse
import logging
import shutil
import pandas as pd
from utils.xml2csv import convert_xml2csv

logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
console = logging.StreamHandler()
console.setFormatter(formatter)

logger.addHandler(console)


class Bbox(object):
    def __init__(self, x_min, y_min, x_max, y_max, score, label, image_w, image_h):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.score = score
        self.label = label
        self.image_w = int(image_w)
        self.image_h = int(image_h)

    def __str__(self):
        return '(Bbox: x_min:%s, y_min:%s, x_max:%s, y_max:%s, score:%s, label:%s, image_w:%s, image_h:%s)' % (self.x_min,
                                                                                                               self.y_min,
                                                                                                               self.x_max,
                                                                                                               self.y_max,
                                                                                                               self.score,
                                                                                                               self.label,
                                                                                                               self.image_w,
                                                                                                               self.image_h)


def iou(bbox1, bbox2):
    x1 = max(bbox1.x_min, bbox2.x_min)
    y1 = max(bbox1.y_min, bbox2.y_min)
    x2 = min(bbox1.x_max, bbox2.x_max)
    y2 = min(bbox1.y_max, bbox2.y_max)

    bbox1_w = bbox1.x_max - bbox1.x_min
    bbox1_h = bbox1.y_max - bbox1.y_min

    bbox2_w = bbox2.x_max - bbox2.x_min
    bbox2_h = bbox2.y_max - bbox2.y_min

    over_w = x2 - x1
    over_h = y2 - y1
    over_w = over_w if over_w > 0 else 0
    over_h = over_h if over_h > 0 else 0

    over_area = over_w * over_h
    iou_value = over_area / (bbox1_w * bbox1_h + bbox2_w * bbox2_h - over_area)
    return iou_value


def nms_matched_boxes(matched_boxes, threshold):
    bounding_boxes = [(x.x_min, x.y_min, x.x_max, x.y_max) for x in matched_boxes]
    bounding_classes = [x.label for x in matched_boxes]
    confidence_score = [x.score for x in matched_boxes]

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_class = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_class.append(bounding_classes[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_class


def load_class_index(class_index_map_path):
    logger.debug('load classify index.')
    classify_index = {}
    with open(class_index_map_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip().split(',')
            classify_index[key] = int(value)
    return classify_index


def load_index_class(class_index_map_path):
    logger.debug('load classify index.')
    index_classify = {}
    with open(class_index_map_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip().split(',')
            index_classify[value] = key
    return index_classify


def generate_test_data_info(xml_path_dir, test_data_info_path):
    convert_xml2csv(os.path.dirname(xml_path_dir), test_data_info_path)
    logger.debug('generate test detail information csv file.')


def csv_to_groundtruth(detail_info, class2index):
    detail_info_dict = {}
    bbox_nums = 0
    if os.path.exists(detail_info):
        with open(detail_info, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                filename, width, height, cls, x_min, y_min, x_max, y_max = line.strip().split(',')[:8]

                if cls not in class2index:
                    continue
                bbox_temp = Bbox(x_min, y_min, x_max, y_max, 100, class2index[cls], width, height)
                if filename not in detail_info_dict:
                    detail_info_dict[filename] = [bbox_temp]
                else:
                    detail_info_dict[filename].append(bbox_temp)
                bbox_nums += 1
    return detail_info_dict, bbox_nums


def csv_to_predict(predict_info, class2index):
    predict_info_dict = {}
    bbox_nums = 0
    if os.path.exists(predict_info):
        with open(predict_info, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                #print(line)
                filename, width, height, cls, x_min, y_min, x_max, y_max, score = line.strip().split(',')[:9]

                if cls not in class2index:
                    logger.warning("{} not correct label!".format(cls))
                    continue
                bbox_temp = Bbox(x_min, y_min, x_max, y_max, float(score), class2index[cls], width, height)
                if filename not in predict_info_dict:
                    predict_info_dict[filename] = [bbox_temp]
                else:
                    predict_info_dict[filename].append(bbox_temp)
                bbox_nums += 1
    return predict_info_dict, bbox_nums


def model_precision_recall(detail_info, predict_info, class2index_path, output, check_images=False, **kwargs):

    if not os.path.exists(output):
        os.makedirs(output)

    ts = time.time()
    class2index = load_class_index(class2index_path)

    groundtruth_info_dict, gt_bbox_nums = csv_to_groundtruth(detail_info, class2index)
    predict_info_dict, predict_bbox_nums = csv_to_predict(predict_info, class2index)
    logger.warning("predict_info: {}.".format(predict_info))
    logger.warning("Ground Truth: {} pictures; Predicted: {} pictures.".format(len(groundtruth_info_dict), len(predict_info_dict)))
    logger.warning("Ground Truth: {} boxes; Predicted: {} boxes.".format(gt_bbox_nums, predict_bbox_nums))

    processing_data_time = time.time() - ts
    logger.debug('Processing test data cost {}ms.'.format(processing_data_time * 1000))

    iou_threshold = kwargs["iou_threshold"]
    score_threshold = kwargs['score_threshold']
    nms_iou_threshold = kwargs['nms_iou_threshold']

    obj_metric = Metric(iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                        nms_iou_threshold=nms_iou_threshold)
    obj_metric.compute_precision_recall_all(predict_info_dict, groundtruth_info_dict)

    obj_metric.predict_boxes_in_file = predict_bbox_nums
    obj_metric.gt_boxes_in_file = gt_bbox_nums

    obj_metric.details()

    format_filename = 'result-{}.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    with open(os.path.join(output, format_filename), 'a', encoding='utf-8') as w:
        w.write("###{}\n".format(predict_info))

        w.write('###box_precision, box_recall, category_precision, category_recall\n')
        w.write("###{}, {}, {}, {}\n".format(obj_metric.box_precision,
                                          obj_metric.box_recall,
                                          obj_metric.label_precision,
                                          obj_metric.label_recall))
        w.write("###测试了 {} 张图片\n###识别率为： {} \n\n".format(obj_metric.predict_nums,
                                                     obj_metric.accuracy))

    total_time = time.time() - ts
    logger.error('Testing cost {:.4f}s, then finished!\n'.format(total_time))

    if check_images:
        obj_metric.split_output_images(output)

    return True


# def analysis_score_threshold(images, annotations, detail_info, predict_info, output, threshold=(30, 100, 5)):
#     start, stop, step = threshold
#     with open('analysis_score_threshold.csv', 'a', encoding='utf-8') as w:
#         w.write('score_threshold,box_precision,box_recall,category_precision,category_recall\n')
#         for score_threshold in range(start, stop, step):
#             score_threshold = score_threshold * 0.01
#             print('current score_threshold is ', score_threshold)
#             _, box_precision, box_recall, label_precision, label_recall = model_precision_recall(annotations,
#                                                                                                  detail_info,
#                                                                                                  predict_info, output,
#                                                                                                  False,
#                                                                                                  score_threshold)
#
#             w.write("{},{},{},{},{}\n".format(threshold, box_precision, box_recall, label_precision, label_recall))
#
#     return True


def split_file_by_class(filename, output, flag):
    file_list = []
    df = pd.read_csv(filename)
    for name, grouped in df.groupby('class'):
        print(filename)
        print(name)
        print(flag)
        per_file = os.path.join(output, flag + "_" + name + ".csv")
        grouped.to_csv(per_file, index=False)
        file_list.append(per_file)
    return file_list


class Metric(object):
    def __init__(self, iou_threshold=0.7, score_threshold=0.5, nms_iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.label_precision = 0
        self.label_recall = 0
        self.box_precision = 0
        self.box_recall = 0
        self.predict_boxes_in_file = 0
        self.gt_boxes_in_file = 0
        self.predict_nums = 0
        self.ground_truth_nums = 0
        self.nms_filter_predict_box_nums = 0
        self.error_files = []
        self.correct_files = []
        self.sample_fp_nums = 0
        self.sample_tp_nums = 0
        self.sample_fn_nums = 0
        self.filter_by_score_threshold_boxes = 0
        self.accuracy = 0

    def details(self):
        logger.info('box_precision: {:.4f}, box_recall: {:.4f}, category_precision: {:.4f}, category_recall: {:.4f}.'.format(
            self.box_precision, self.box_recall, self.label_precision, self.label_recall))
        logger.info("预测文件有 {} 框,nms过滤 {} 框,参与计算 {} 框,GroundTruth: {} 框.".format(self.predict_boxes_in_file,
                                                                       self.nms_filter_predict_box_nums,
                                                                       self.predict_nums,
                                                                       self.ground_truth_nums))
        logger.info("识别率为： {:.4f}.".format(self.accuracy))
        logger.info("sample_fp: {}, sample_tp: {}, sample_fn: {}.".format(self.sample_fp_nums, self.sample_tp_nums, self.sample_fn_nums))
        logger.info("low score boxes nums: {}.".format(self.filter_by_score_threshold_boxes))

    def split_output_images(self, output_images):
        if not os.path.exists(output_images):
            logger.error("{} not exists!".format(output_images))
            return None
        error_dir = os.path.join(output_images, 'error')
        correct_dir = os.path.join(output_images, 'correct')

        if not os.path.exists(error_dir):
            os.mkdir(error_dir)
        if not os.path.exists(correct_dir):
            os.mkdir(correct_dir)

        for filename in self.error_files:
            src_img = os.path.join(output_images, filename)
            if os.path.exists(src_img):
                shutil.move(src_img, os.path.join(error_dir, filename))
        for filename in self.correct_files:
            src_img = os.path.join(output_images, filename)
            if os.path.exists(src_img):
                shutil.move(src_img, os.path.join(correct_dir, filename))

    def match_box(self, box1, box2):
        iou_score = iou(box1, box2)
        if iou_score > self.iou_threshold:
            return iou_score, box1.label == box2.label
        else:
            return -1, False

    def compute_precision_recall_all(self, predict, ground_truth):
        """
        True Positive （TP）被模型预测为正样本，实际为正样本；
        False Positive（FP）被模型预测为正样本，实际为负样本；
        True Negative （TN）被模型预测为负样本，实际为负样本；
        False Negative（FN）被模型预测为负样本，实际为正样本；

        accuracy  = （TP+TN）/(TP+FP+TN+FN)
        precision = TP/(TP+FP)
        recall    = TP/(TP+FN)

        :param predict:
        :param ground_truth:
        :param threshold:
        :return:
        """
        box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative = [0] *6
        for gt_key, gt_value in ground_truth.items():

            box_fp, box_tp, box_fn, label_fp, label_tp, label_fn = [0] * 6
            if gt_key in predict:
                box_fp, box_tp, box_fn, label_fp, label_tp, label_fn = self.compute_precision_recall_per_image(predict[gt_key], gt_value)
            else:
                # 漏检
                box_fn = len(gt_value)
                label_fn = box_fn
                logger.warning("{} predict nothing, ground-truth length is {}.".format(gt_key, len(gt_value)))

            # pop key from dict
            predict.pop(gt_key, None)

            # count files
            if label_fp > 0 or label_fn > 0:
                self.error_files.append(gt_key)
                # print(str(predict[gt_key]))
                # print()
                # print(str(gt_value))
            else:
                self.correct_files.append(gt_key)

            box_false_positive += box_fp
            box_true_positive += box_tp
            box_false_negative += box_fn
            label_false_positive += label_fp
            label_true_positive += label_tp
            label_false_negative += label_fn
            logger.debug("box_fp: {}, box_tp: {}, box_fn:{};label_fp: {}, label_tp: {}, label_fn:{}.".format(
                box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative
            ))

        if len(predict) > 0:
            for predict_key, predict_value in predict.items():
                box_false_positive += len(predict_value)
                label_false_positive += len(predict_value)
                logger.error("{} has not ground-truth".format(predict_key))

        box_n_precision = box_true_positive + box_false_positive
        self.box_precision = 0 if box_n_precision == 0 else box_true_positive / box_n_precision
        box_n_recall = box_true_positive + box_false_negative
        self.box_recall = 0 if box_n_recall == 0 else box_true_positive / box_n_recall

        label_n_precision = label_true_positive + label_false_positive
        self.label_precision = 0 if label_n_precision == 0 else label_true_positive / label_n_precision
        label_n_recall = label_true_positive + label_false_negative
        self.label_recall = 0 if label_n_recall == 0 else label_true_positive / label_n_recall

        logger.info('box_false_positive: {}, box_true_positive: {}, box_false_negative:{}.'.format(
            box_false_positive, box_true_positive, box_false_negative
        ))
        logger.info('label_false_positive: {}, label_true_positive: {}, label_false_negative:{}.'.format(
            label_false_positive, label_true_positive, label_false_negative
        ))
        # label_true_negative is zero.
        if label_true_positive > 0:
            self.accuracy = label_true_positive / (label_true_positive + label_false_positive + label_false_negative)
        logger.info('label accuracy is: {}'.format(self.accuracy))

        self.sample_fn_nums = label_false_negative
        self.sample_tp_nums = label_true_positive
        self.sample_fp_nums = label_false_positive

    def compute_precision_recall_per_image(self, predicts, ground_truth):
        box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative = [0] * 6

        logger.debug("predict boxes:{}; gt boxes:{}.".format(len(predicts), len(ground_truth)))

        if len(predicts) <= 0:
            box_false_negative = len(ground_truth)
            label_false_negative = box_false_negative
            return box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative
        if len(ground_truth) <= 0:
            if isinstance(predicts, dict):
                box_false_positive = len(predicts['detection_boxes'])
            else:
                box_false_positive = len(predicts)
            label_false_positive = box_false_positive

            return box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative

        predict_bboxes = self.output_to_bboxes(predicts, ground_truth[0].image_w, ground_truth[0].image_h)

        # sort by score
        predict_bboxes = sorted(predict_bboxes, key=lambda x: x.score, reverse=True)
        split_mark = 0
        for index, bbox in enumerate(predict_bboxes):
            if bbox.score > self.score_threshold:
                split_mark += 1
            else:
                break
        predict_bboxes = predict_bboxes[:split_mark]
        self.filter_by_score_threshold_boxes += len(predict_bboxes) - split_mark

        self.ground_truth_nums += len(ground_truth)
        self.predict_nums += len(predict_bboxes)

        for gt_box in ground_truth:
            logger.debug("gt_box: {}.".format(gt_box.__str__()))
            find_box_flag = False
            find_label_flag = False

            matched_index = -1
            matched_iou_max = 0

            # find matched box in predict
            for index, predict_box in enumerate(predict_bboxes):
                matched_box_iou, matched_label = self.match_box(predict_box, gt_box)
                if matched_box_iou > matched_iou_max:
                    matched_iou_max = matched_box_iou
                    find_box_flag = True
                    matched_index = index
                if matched_label:
                    find_label_flag = True
                    matched_index = index
                    logger.debug("matched predict box: {}.".format(predict_box.__str__()))
                    break

            if matched_index != -1:
                predict_bboxes.pop(matched_index)

            if find_box_flag and find_label_flag:
                box_true_positive += 1
                label_true_positive += 1
            elif find_box_flag:
                box_true_positive += 1
                label_false_negative += 1
            else:
                box_false_negative += 1
                label_false_negative += 1
        # 误识别
        error_recog = len(predict_bboxes)
        if error_recog:
            box_false_positive += error_recog
            label_false_positive += error_recog

        return box_false_positive, box_true_positive, box_false_negative, label_false_positive, label_true_positive, label_false_negative

    def output_to_bboxes(self, predicts, image_w, image_h):
        predict_bboxes = []
        if isinstance(predicts, dict):
            for boxes, classes, scores in zip(predicts['detection_boxes'],
                                              predicts['detection_classes'],
                                              predicts['detection_scores']):
                ymin, xmin, ymax, xmax = boxes
                image_h = int(image_h)
                image_w = int(image_w)
                ymin, ymax = ymin * image_h, ymax * image_h
                xmin, xmax = xmin * image_w, xmax * image_w
                bbox_temp = Bbox(int(xmin), int(ymin), int(xmax), int(ymax), scores, classes, image_w, image_h)
                predict_bboxes.append(bbox_temp)
        else:
            predict_bboxes = predicts

        # before nms
        #print('***Before NMS, predict_bboxes length is ', len(predict_bboxes))
        # image_path = "data/stage1/JPEGImages/TV_CAM_%E8%AE%BE%E5%A4%87_20200725_105512.235.jpg"
        # test_nms_in_matched_boxes(image_path, predict_bboxes, threshold=0.5)

        self.nms_filter_predict_box_nums += len(predict_bboxes)

        picked_boxes, picked_score, picked_class = nms_matched_boxes(predict_bboxes, self.nms_iou_threshold)
        predict_bboxes = [
            Bbox(box[0], box[1], box[2], box[3], score, label, image_w, image_h) for
            box, score, label in zip(picked_boxes, picked_score, picked_class)]
        # bbox_temp = Bbox(x_min, y_min, x_max, y_max, 0, class2index[cls], width, height)
        #print('***After NMS, predict_bboxes length is ', len(predict_bboxes))
        # end nms
        self.nms_filter_predict_box_nums -= len(picked_boxes)

        return predict_bboxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Milk Package Detect.')
    parser.add_argument('--test_data_dir', type=str,
                        help='test data directory path.')
    parser.add_argument('--class2index_path', type=str,
                        help='class2index path.')
    parser.add_argument('--output', type=str,
                        help='Testing result dir.')
    parser.add_argument('--split_output', type=int, default=0,
                        help='split output.')
    parser.add_argument('--iou_threshold', type=float,
                        help='iou_threshold.')
    parser.add_argument('--score_threshold', type=float,
                        help='score_threshold.')
    parser.add_argument('--nms_iou_threshold', type=float,
                        help='nms_iou_threshold.')
    args = parser.parse_args()

    test_data_dir = args.test_data_dir

    annotation_dir_path = os.path.join(test_data_dir, 'Annotations')
    jpeg_dir_path = os.path.join(test_data_dir, 'JPEGImages')

    ground_truth_detail_info = os.path.join(test_data_dir, 'groundtruth_data_details.csv')
    if not os.path.exists(ground_truth_detail_info):
        generate_test_data_info(annotation_dir_path, ground_truth_detail_info)
        logger.debug('Convert annotations to ground-truth csv file.')

    predict_detail = os.path.join(test_data_dir, 'predict_data_details.csv')

    threshold_dict = {'iou_threshold': args.iou_threshold, 'score_threshold': args.score_threshold,
                      'nms_iou_threshold': args.nms_iou_threshold}
    
    temp_info_dir = "temp_info_dir"
    if not os.path.exists(temp_info_dir):
        os.mkdir(temp_info_dir)
    
    data_detail_split = split_file_by_class(ground_truth_detail_info, temp_info_dir, "groundtruth")
    predict_detail_split = split_file_by_class(predict_detail, temp_info_dir, 'predict')

    split_output = True if args.split_output == 1 else False

    for gt_per in data_detail_split:
        predict_per = gt_per.replace("groundtruth", "predict")
        test_state = model_precision_recall(gt_per, predict_per, args.class2index_path, args.output, split_output, **threshold_dict)
    # test_state = analysis_score_threshold(annotation_dir_path, data_detail, predict_detail, args.output, threshold=(30, 100, 5))

    test_state = model_precision_recall(ground_truth_detail_info, predict_detail, args.class2index_path, args.output, split_output, **threshold_dict)

    if test_state:
        logger.warning('Testing successfully!')
    else:
        logger.error('Testing error!')


