# 3D_HandPose

This repository implements a realtime 3D hand posture estimation pipeline running on Jetson platform using a [**Azure Kinect** camera](https://azure.microsoft.com/en-us/services/kinect-dk/).<br/>


There are 2 stages to our pipeline

* ## [CenterNet Bounding Box](#centernet_bounding_box)
* ## [A2J Posture Detection](#a2j_posture_detection)
* ## [Run inference](#run_infrence)

<a name="centernet_bounding_box"></a>
## CenterNet Bounding Box

The first stage will localize the hand using a fusion of infrared and depth image.<br/>


<a name="a2j_posture_detection"></a>
## A2J Posture Detection

The second stage would perform 3D hand posture estimation on the region of intrest selected by the previous step.<br/>


<a name="run_infrence"></a>
## Run inference

- Run realtime inference on a jetson platform.
    ```bash
    cd pipeline
    python3 azure_kinect.py
    
    # Optional for faster inference
    python3 azure_kinect.py --trt True # for optimizing the models with TensorRT fp16  
    ```
