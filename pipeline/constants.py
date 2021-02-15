
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)

CENTERNET_MODEL_PATH = os.path.join(ROOT_PATH, "checkpoint/CenterNet")
A2J_MODEL_PATH = os.path.join(ROOT_PATH, "checkpoint/A2J")
#############################################
############# CenterNet SETUP ###############
#############################################
CENTERNET_MODEL_NAME = "ResnetCenterNet"

# Setup the data to be used for training (Depth images/ fusion of IR and Depth images)
CENTERNET_DATA_LOADER_SWITCHER = {
    "depth": False,
    "fused": True,
}
CENTERNET_DATA_LOADER = [[elem[0] for elem in CENTERNET_DATA_LOADER_SWITCHER.items() if elem[1]][0]] [0]

# Setup the heatmap loss MSE/Logistic loss
CENTERNET_LOSS_SWITHCER = {
    "MSE": False,
    "Logistic": True,
}
CENTERNET_LOSS = [[elem[0] for elem in CENTERNET_LOSS_SWITHCER.items() if elem[1]][0]] [0]

CENTERNET_IMG_SHAPE = (320, 320)

CENTERNET_NUM_CLASSES = 1
CENTERNET_STRIDE = 2

THRESHOLD_ACC = 0.3

INPUT_IMG_SIZE = (320, 320)

#############################################
################# A2J SETUP #################
#############################################
DATASET = "NYU" # "Personal", "NYU"

DATA_SEGMENT = "1" # ALL, 1
# List of availiblke backbones set the one you wantto use to true and all else to false
A2J_BACKBONE_NAME = {
    "resnet18": False,
    "resnet34": False,
    "resnet50": True,
    "resnet101": False,
    "resnet152": False,
    "mobilenet": False,
}

A2J_TARGET_SIZE = (176, 176)
DEPTH_THRESHOLD = 180
A2J_STRIDE = 16
NUM_JOINTS = 16 # 14, 16, 36, 21
