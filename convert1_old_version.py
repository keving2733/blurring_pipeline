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
    bag =rosbag.Bag(args.output_bag, 'w')
    idx = 0
    for i in range(len(args.bags)):
        # print ("Saving every %d images from %s on topic %s into) %s" % (args.sample, args.bags[i], args.topics[i], args.output_dir))
        bag = rosbag.Bag(args.bags[i], "r")
        # count = 0
        for topic_name, msg, t in bag.read_messages(topics=args.topics):
            # if count % args.sample == 0:
            cv_img = None
            if args.compressed:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg)
            
            else:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            blurred_image = detect_and_blur(cv_img)
            
            blurred_message = bridge.cv2_to_imgmsg(blurred_image, encoding="passthrough")

            bag.write(topic_name,blurred_message,t)
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

def detect_and_blur():

    # Haar Cascade Classifier

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    img = cv2.imread('/home/keving07/Downloads/testbw.jpg')
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    # Draw rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # # Display the output
    # cv2.imshow('img', img)
    # cv2.waitKey()

    # # DNN Classifier
    # # image1 = '/home/keving07/Downloads/test_multi3.jpg'
    # face_detection_model = '/home/keving07/opencv_zoo/models/face_detection_yunet/face_detection_yunet_2022mar.onnx'
    # score_threshold = 0.9
    # nms_threshold = 0.3
    # top_k = 5000
    # scale = 1
    
    # detector = cv2.FaceDetectorYN.create(
    #     face_detection_model,
    #     "",
    #     (320, 320),
    #     score_threshold,
    #     nms_threshold,
    #     top_k
    # )
    
    # tm = cv2.TickMeter()
    # # If input is an image
    # # if image1 is not None:
    # # img1 = cv2.imread(cv2.samples.findFile(image1))
    # img1Width = int(img1.shape[1]*scale)
    # img1Height = int(img1.shape[0]*scale)
    # img1 = cv2.resize(img1, (img1Width, img1Height))
    # tm.start()
        
    # detector.setInputSize((img1Width, img1Height))
    # faces = detector.detect(img1)
    
    # tm.stop()
    # assert faces[1] is not None, 'Cannot find a face in {}'.format(img1)
    # Draw results on the input image
    # visualize(img1, faces, tm.getFPS())
    # Save results if save is true

    # print('Results saved to result.jpg\n')
        
        # Visualize results in a new window

    # Blurring for Haar Cascade Classifier
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]

        # apply gaussian blur to face rectangle
        roi = cv2.GaussianBlur(roi, (51, 51), 50)

        # add blurred face on original image to get final image
        img[y:y + roi.shape[0], x:x + roi.shape[1]] = roi

    # # Display the output
    # for idx, face in enumerate(faces[1]):
    #     coords = face[:-1].astype(np.int32)
    #     x0,y0 = (coords[0], coords[1])
    #     x1,y1 = (coords[0]+coords[2], coords[1]+coords[3])

    #     roi = img1[y0:y1, x0:x1]
    #     # apply gaussian blur to face rectangle
    #     roi = cv2.GaussianBlur(roi, (51, 51), 50)

    #     # add blurred face on original image to get final image
    #     img1[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]] = roi
    # cv2.imshow('Blur Face', img1)
    cv2.imwrite('blurred_result.jpg', img)
    # cv2.waitKey(0)
    # return img1 


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    detect_and_blur()
    # extract_and_write_imgs()

#python3 /home/keving07/Downloads/convert_1.py --bags /home/keving07/Downloads/acl_jackal2_test3.bag --output_dir /home/keving07/Downloads/test_folder --topics /acl_jackal2/forward/infra2/image_rect_raw/compressed /acl_jackal2/forward/infra1/image_rect_raw/compressed /acl_jackal2/forward/infra2/image_rect_raw/compressed --sample 1
