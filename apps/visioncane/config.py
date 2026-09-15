DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_YOLO_WEIGHTS = "assets/yolov8n.pt"
DEFAULT_CAM_INDEX = 0
PROC_SHORT_SIDE = 512
DEVICE = "cuda"

FOV_DEG = 175.0
INVERT_DEFAULT = True

# YOLO
YOLO_CONF = 0.30
MAX_DETS = 30

# Depth -> proximity percentile clipping
PCTL_LOW = 5
PCTL_HIGH = 95

# Border handling
BORDER_IGNORE_FRAC = 0.10
BORDER_PENALTY_FRAC = 0.12

# Bottom-row bias (tables, low obstacles)
BOTTOM_GAMMA = 2.2
BOTTOM_MIN_MULT = 1.0
BOTTOM_MAX_MULT = 2.2

# "At least close" gate — no beep if nothing near
CLOSE_COLOR_THRES = 0.90
GLOBAL_CLOSE_MIN_FRAC = 0.008
BBOX_CLOSE_MIN_FRAC = 0.05

# Target locking
IOU_LOCK_THRES = 0.20
MAX_LOST_FRAMES = 8

# Depth blob fallback (used when YOLO finds nothing)
CLOSE_QUANTILE = 0.92
MIN_AREA_FRAC = 0.020
MORPH_KERNEL = 9

# Audio output parameters
MIN_INTERVAL = 0.07
MAX_INTERVAL = 0.30
MIN_VOL = 0.05
MAX_VOL = 0.95
PITCH_FAR = 1200.0
PITCH_NEAR = 1200.0
