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
        fd_score_threshold = 0.8
        fd_nms_threshold = 0.3
        fd_top_k = 500
        lpd_score_threshold = 0.8
        lpd_nms_threshold = 0.3
        lpd_top_k = 500

        self.face_detector = cv2.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            fd_score_threshold,
            fd_nms_threshold,
            fd_top_k,
            cv2.dnn.DNN_BACKEND_CUDA,
            cv2.dnn.DNN_TARGET_CUDA
        )

        self.lp_detector = cv2.FaceDetectorYN.create(
            licence_plate_model,
            "",
            (320, 320),
            lpd_score_threshold,
            lpd_nms_threshold,
            lpd_top_k,
            cv2.dnn.DNN_BACKEND_CUDA,
            cv2.dnn.DNN_TARGET_CUDA
        )

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
        plates = self.lp_detector.detect(img)

        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x_start, y_start = (coords[0], coords[1])
                x_end, y_end = (coords[0]+coords[2], coords[1]+coords[3])

                roi = img[y_start:y_end, x_start:x_end]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                if roi.shape[0] > imgHeight / 6 or roi.shape[1] > imgWidth / 6:
                    continue

                # apply gaussian blur to face rectangle
                roi = cv2.GaussianBlur(roi, (51, 51), 50)

                # add blurred face on original image to get final image
                try:
                    img[y_start:y_start + roi.shape[0],
                        x_start:x_start + roi.shape[1]] = roi
                except ValueError:
                    pass

        if plates[1] is not None:
            for idx, plate in enumerate(plates[1]):
                coords = plate[:-1].astype(np.int32)
                x_start, y_start = (coords[0], coords[1])
                x_end, y_end = (coords[0]+coords[2], coords[1]+coords[3])

                roi = img[y_start:y_end, x_start:x_end]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                if roi.shape[0] > imgHeight / 6 or roi.shape[1] > imgWidth / 6:
                    continue

                # apply gaussian blur to face rectangle
                roi = cv2.GaussianBlur(roi, (51, 51), 50)

                # add blurred face on original image to get final image
                try:
                    img[y_start:y_start + roi.shape[0],
                        x_start:x_start + roi.shape[1]] = roi
                except ValueError:
                    pass

        if gray_image:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            print('error with making image')
        return img
