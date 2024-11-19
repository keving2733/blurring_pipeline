# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from cv_bridge import CvBridge
import os
import argparse
import rosbag
import numpy as np
import yaml
import click
import cv2

from face_detect_utils import DetectBlurPipeline
from sensor_msgs.msg import Image, CompressedImage

def blur_bag(source, topics, blur_topics, output_robot, output, fd_onnx_file, lp_onnx_file):
    """Extract images from bag and write to new bag
    """
    print("Blurring rosbag: {}".format(source))
    processor = DetectBlurPipeline(fd_onnx_file, lp_onnx_file)
    bridge = CvBridge()
    outbag = rosbag.Bag(output, 'w')
    previous_faces = []
    inbag = rosbag.Bag(source, "r")
    for topic_name, msg, t in inbag.read_messages(topics=topics):
        topic_name_list = topic_name.split("/")
        topic_name_list[1] = output_robot
        new_topic = "/".join(topic_name_list)

        if topic_name not in blur_topics:
            outbag.write(new_topic, msg, t)
            continue

        compressed = (msg._type == "sensor_msgs/CompressedImage")
        if compressed:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg)
        else:
            cv_img = bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

        # Blur image
        blurred_image = processor.detect_and_blur(cv_img)

        if compressed:
            blurred_message = bridge.cv2_to_compressed_imgmsg(
                blurred_image)

        else:
            blurred_message = bridge.cv2_to_imgmsg(
                blurred_image, encoding="passthrough")
        blurred_message.header = msg.header

        outbag.write(new_topic, blurred_message, t)
    outbag.close()
    return


def main():
    parser = argparse.ArgumentParser(
        description="Detect and blur faces from select topics of rosbag.")
    parser.add_argument("config", help="input config file")
    args = parser.parse_args()
    with open(args.config, 'r') as config_file:
        configs = yaml.safe_load(config_file)
        # print(topics, blur_topics)
        input_bags = configs["bag_path"]
        input_robots = configs["robot_name"]
        output_robots = configs["out_robot_name"]
        output_bags = configs["output_path"]
        for i in range(len(input_bags)):
            bag_path = input_bags[i]
            robot_name = input_robots[i]
            topics = ["/" + robot_name + "/" + t for t in configs["topics"]]
            blur_topics = ["/" + robot_name + "/" +
                       t for t in configs["blur_topics"]]
            output_robot = output_robots[i]
            output_path = output_bags[i]
            blur_bag(bag_path, topics, blur_topics, output_robot, output_path,
                     configs["face_onnx_file"], configs["license_plate_onnx_file"])


if __name__ == '__main__':
    main()
