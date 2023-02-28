# blurring_pipeline

This is a blurring pipeline for faces in rosbag files. Two different algorithms can be used. There is the Haar Cascade and using a DNN. The DNN is the most accurate
that requires other repositorys and packages to be installed.

First clone this respository: https://github.com/opencv/opencv_zoo to your machine, follow the instructions at the link.

You must also install open-cv contrib using pip install with this command:

pip install opencv-contrib-python

In the "detect_and_blur" function on line 89, change the face detection model path to the path to the model on your machine. This is located in the opencv_zoo
repository in models>face_detection_yunet and the file is named "face_detection_yunet_2022mar.onnx.

Then, you should be good to run the script from the terminal, passing in arguments using argparse. See lines 18-25 to check what parameters are required.
