# blurring_pipeline

This is a blurring pipeline for faces in rosbag files. Two different algorithms can be used. There is the Haar Cascade and using a DNN. The DNN is the most accurate
that requires other repositorys and packages to be installed.

First clone this respository: https://github.com/opencv/opencv_zoo to your machine, follow the instructions at the link.

You must also install open-cv contrib using pip install with this command:

pip install opencv-contrib-python

In the "detect_and_blur" function on line 89, change the face detection model path to the path to the model on your machine. This is located in the opencv_zoo
repository in models>face_detection_yunet and the file is named "face_detection_yunet_2022mar.onnx.

Then, you should be good to run the script from the terminal, passing in arguments using argparse. See lines 18-25 to check what parameters are required.

## Docker Instructions

Install docker and Nvidia-docker2.
Modify your `/etc/docker/daemon.json` to look like:
```
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "/usr/bin/nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
```

Build image
```bash
docker build -t blurring_pipeline:latest .
```

Run image as container
```bash
docker run --name blurring_pipeline -d -i -t -v /media/yunchang/YUN_SanDisk2:/data blurring_pipeline /bin/bash
```

Enter bash
```bash
docker exec -it blurring_pipeline /bin/bash
```