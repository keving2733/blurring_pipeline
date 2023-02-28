# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os
import argparse
import rosbag
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def extract_and_write_imgs():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(
        description="Save images from image topic in a rosbag to file.")
    parser.add_argument("-b", "--bags", nargs='+', default=[], help="rosbags")
    parser.add_argument("-ob", "--output_bag", help="path for output bag.")
    parser.add_argument("-t", "--topics", nargs='+', default=[], help="topics")
    # parser.add_argument("-s", "--sample", type=int,
    #                     default=1, help="save every nth image.")
    parser.add_argument("-c", "--compressed", type=bool,
                        default=False, help="compressed image or not.")

    args = parser.parse_args()
    # parser.bags = ['/home/keving07/Downloads/acl_jackal2_test3.bag']
    # parser.output_dir = os.makedirs('/home/keving07/Downloads/jackal2_test3_imgs')
    bridge = CvBridge()
    # if len(args.bags) != len(args.topics):
    #     print("Number of rosbags does not match number of topics specified. Please specify topic name for each bag. ")
    #     return
    bag = rosbag.Bag(args.output_bag, 'w')
    idx = 0
    previous_faces = []
    for i in range(len(args.bags)):
        # print ("Saving every %d images from %s on topic %s into) %s" % (args.sample, args.bags[i], args.topics[i], args.output_dir))
        bag = rosbag.Bag(args.bags[i], "r")
        # count = 0
        for topic_name, msg, t in bag.read_messages(topics=args.topics):
            # if count % args.sample == 0:
            if args.compressed:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg)
            
            else:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            blurred_image = detect_and_blur(cv_img)

            if args.compressed:
                blurred_message = bridge.cv2_to_compressed_imgmsg(blurred_image)

            else:
                blurred_message = bridge.cv2_to_imgmsg(blurred_image, encoding="passthrough")
            # print('ros message: ')
            # print(type(msg))
            # print('   time: ')
            # print(type(t))
            try:
                bag.write(topic_name,blurred_message,t)
            except:
                bag.write(topic_name,msg,t)
            idx += 1

            # count += 1

        bag.close()

    return

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_and_blur(raw_img):

    #path to face detection model
    face_detection_model = '/home/keving07/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2022mar.onnx'
    score_threshold = 0.9
    nms_threshold = 0.3
    top_k = 5000
    scale = 1
    
    detector = cv2.FaceDetectorYN.create(
        face_detection_model,
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k
    )
    gray_image = False
    tm = cv2.TickMeter()
    # If input is an image
    # if image1 is not None:
    # img = cv2.imread(cv2.samples.findFile(image1))
    if len(raw_img.shape) == 2:
        img = cv2.cvtColor(raw_img,cv2.COLOR_GRAY2RGB)
        gray_image = True
    else:
        img = raw_image 
    imgWidth = int(img.shape[1]*scale)
    imgHeight = int(img.shape[0]*scale)
    img = cv2.resize(img, (imgWidth, imgHeight))
    tm.start()
       
    detector.setInputSize((imgWidth, imgHeight))
    faces = detector.detect(img)
    
    tm.stop()
    # assert faces[1] is not None, 'Cannot find a face in {}'.format(img)
    # Draw results on the input image, drawing a rectangle around the
    # Visualize results in a new window 
    # visualize(img, faces, tm.getFPS())
    # Save results if save is true


    # Display the output
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x_start,y_start = (coords[0], coords[1])
            x_end,y_end = (coords[0]+coords[2], coords[1]+coords[3])

            roi = img[y_start:y_end, x_start:x_end]
            # apply gaussian blur to face rectangle
            roi = cv2.GaussianBlur(roi, (51, 51), 50)

            # add blurred face on original image to get final image
            img[y_start:y_start + roi.shape[0], x_start:x_start + roi.shape[1]] = roi
    if gray_image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        print('error with making image')
    return img 


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # detect_and_blur('/home/keving07/Downloads/testbw.jpg')
    extract_and_write_imgs()

# python3 /home/keving07/Downloads/blurring_pipeline/rosbag_multiface_blurr.py --bags /home/keving07/Downloads/acl_jackal2_test3.bag -ob /home/keving07/Downloads/test_folder --topics /acl_jackal2/forward/infra2/image_rect_raw/compressed /acl_jackal2/forward/infra1/image_rect_raw/compressed -c true
