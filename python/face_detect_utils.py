# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os
import argparse
import rosbag
import numpy as np

from lpd_yunet import LPD_YuNet


class DetectBlurPipeline:
    def __init__(self, fd_onnx_file, lp_onnx_file):
        # path to face detection model
        face_detection_model = fd_onnx_file
        licence_plate_model = lp_onnx_file
        score_threshold = 0.8
        nms_threshold = 0.4
        top_k = 5000
        scale = 1

        self.face_detector = cv2.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )

        self.lp_detector = LPD_YuNet(modelPath=licence_plate_model,
                                     confThreshold=score_threshold,
                                     nmsThreshold=nms_threshold,
                                     topK=top_k,
                                     keepTopK=800,
                                     backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                                     targetId=cv2.dnn.DNN_TARGET_CPU)

    def detect_and_blur(self, raw_img):
        gray_image = False
        if len(raw_img.shape) == 2:
            img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
            gray_image = True
        else:
            img = raw_img
        imgWidth = img.shape[1]
        imgHeight = img.shape[0]
        img = cv2.resize(img, (imgWidth, imgHeight))

        # Detect faces
        self.face_detector.setInputSize((imgWidth, imgHeight))
        faces = self.face_detector.detect(img)

        # Detect license plates
        self.lp_detector.setInputSize([imgWidth, imgHeight])
        plates = self.lp_detector.infer(img)

        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x_start, y_start = (coords[0], coords[1])
                x_end, y_end = (coords[0]+coords[2], coords[1]+coords[3])

                roi = img[y_start:y_end, x_start:x_end]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                # apply gaussian blur to face rectangle
                roi = cv2.GaussianBlur(roi, (51, 51), 50)

                # add blurred face on original image to get final image
                img[y_start:y_start + roi.shape[0],
                    x_start:x_start + roi.shape[1]] = roi

        if plates.shape[0] > 0:
            for plate in plates:
                coords = plate[:-1].astype(np.int32)
                x_start, y_start = (coords[0], coords[1])
                x_end, y_end = (coords[0]+coords[2], coords[1]+coords[3])

                roi = img[y_start:y_end, x_start:x_end]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                # apply gaussian blur to face rectangle
                roi = cv2.GaussianBlur(roi, (51, 51), 50)

                # add blurred face on original image to get final image
                img[y_start:y_start + roi.shape[0],
                    x_start:x_start + roi.shape[1]] = roi

        if gray_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            print('error with making image')
        return img
