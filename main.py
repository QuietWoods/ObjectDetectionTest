# coding: utf-8
import os
import argparse
import object_detection_script
import logging

logger = logging.getLogger("main")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test object detect.')
    parser.add_argument('--model_path', type=str,
                        help='model path.')
    parser.add_argument('--quantized', type=int, default=0,
                        help='quantized model.')
    parser.add_argument('--label_path', type=str,
                        help='label path.')
    parser.add_argument('--class2index_path', type=str,
                        help='class2index path.')
    parser.add_argument('--num_classes', type=int,
                        help='the number of classes.')
    parser.add_argument('--test_data_dir', type=str,
                        help='test data directory path.')
    parser.add_argument('--test_num', type=int, default=10,
                        help='option test number, default value is 10.')
    parser.add_argument('--output', type=str,
                        help='Testing result dir.')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='score threshold.')
    parser.add_argument('--scale_saved_image', type=int, default=1,
                        help='is scale saved image.')
    parser.add_argument('--width', type=int, default=480,
                        help='the width of saved image.')
    parser.add_argument('--height', type=int, default=640,
                        help='the height of saved image.')
    parser.add_argument('--shape_flag', type=int, default=0,
                        help='is brand to shape.')
    args = parser.parse_args()
    print(args)
    quantized = True if args.quantized == 1 else False
    scale_saved_image = True if args.scale_saved_image == 1 else False
    shape_flag = True if args.shape_flag == 1 else False

    test_data_dir = args.test_data_dir

    jpeg_dir_path = os.path.join(test_data_dir, 'JPEGImages')

    if scale_saved_image:
        save_image_size = (args.width, args.height)
    else:
        save_image_size = None

    test_state = object_detection_script.detect_pipeline(jpeg_dir_path,
                                                         args.model_path, args.label_path, args.class2index_path,
                                                         args.num_classes,
                                                         args.score_threshold, args.output, quantized,
                                                         save_image_size, shape_flag, args.test_num)
    if test_state:
        logger.warning('Testing successfully!')
    else:
        logger.error('Testing error!')
