# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os
import argparse
import rosbag
import numpy as np
import yaml
import click
import time

from face_detect_utils import DetectBlurPipeline
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


def blur_video(source, output, fd_onnx_file, lp_onnx_file):
    processor = DetectBlurPipeline(fd_onnx_file, lp_onnx_file)
    input_video = cv2.VideoCapture(source)

    frame_width = 0
    frame_height = 0
    num_frames = 0
    start_time = time.time()

    if not input_video.isOpened():
        print("Error opening input video.")
        return

    blurred_frames = []
    frame_id = 0
    while input_video.isOpened():
        ret, frame = input_video.read()
        if ret:
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            num_frames += 1
            # blur frame
            blurred_image = processor.detect_and_blur(frame)
            blurred_frames.append(blurred_image)
        print(frame_id)
        frame_id += 1
        if frame_id > 5000:
            break

    end_time = time.time()
    input_video.release()

    fps = int(num_frames / (end_time - start_time))
    output_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 5, (frame_height, frame_width))

    for frame in blurred_frames:
        output_video.write(frame)

    output_video.release()


def main():
    parser = argparse.ArgumentParser(
        description="Detect and blur faces from select topics of rosbag.")
    parser.add_argument("-i", "--input", nargs='+',
                        default=[], help="input videos")
    parser.add_argument("-o", "--output", nargs='+',
                        default=[], help="output videos")
    parser.add_argument("fd_onnx", help="path to face detection onnx file")
    parser.add_argument(
        "lp_onnx", help="path to the license plate detection onnx file")
    args = parser.parse_args()

    if len(args.input) != len(args.output):
        print("Number of output does not equal the number of input.")
        return
    for i in range(len(args.input)):
        blur_video(args.input[i], args.output[i], args.fd_onnx, args.lp_onnx)


if __name__ == '__main__':
    main()
