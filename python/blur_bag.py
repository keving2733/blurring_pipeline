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

from face_detect_utils import detect_and_blur
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


def blur_bag(source, topics, blur_topics, output, onnx_file):
    """Extract images from bag and write to new bag
    """
    bridge = CvBridge()
    outbag = rosbag.Bag(output, 'w')
    previous_faces = []
    inbag = rosbag.Bag(source, "r")
    for topic_name, msg, t in inbag.read_messages(topics=topics):
        if topic_name not in blur_topics:
            outbag.write(topic_name, msg, t)
            continue

        compressed = (msg._type == "sensor_msgs/CompressedImage")
        if compressed:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg)
        else:
            cv_img = bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

        # Blur image
        blurred_image = detect_and_blur(cv_img, onnx_file)

        if compressed:
            blurred_message = bridge.cv2_to_compressed_imgmsg(
                blurred_image)

        else:
            blurred_message = bridge.cv2_to_imgmsg(
                blurred_image, encoding="passthrough")

        outbag.write(topic_name, blurred_message, t)
    outbag.close()
    return


def main():
    parser = argparse.ArgumentParser(
        description="Detect and blur faces from select topics of rosbag.")
    parser.add_argument("config", help="input config file")
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        configs = yaml.safe_load(config_file)
        robot_name = configs["robot_name"]
        topics = ["/" + robot_name + "/" + t for t in configs["topics"]]
        blur_topics = ["/" + robot_name + "/" +
                       t for t in configs["blur_topics"]]
        # print(topics, blur_topics)
        blur_bag(configs["bag_path"], topics,
                 blur_topics, configs["output_path"], configs["onnx_file"])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
