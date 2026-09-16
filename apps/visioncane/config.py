# Depth source --------------------------------------------------------------------
# Metric checkpoints return absolute metres, so every alarm threshold below is a real
# distance and the cane stays silent when nothing is genuinely close. Set
# METRIC_DEPTH = False with RELATIVE_MODEL_ID to fall back to the scene-relative
# pipeline (no absolute distances: it ranks what is nearest, but cannot tell whether
# the nearest thing is 1 m or 30 m away).
METRIC_DEPTH = True
METRIC_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
RELATIVE_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_MODEL_ID = METRIC_MODEL_ID if METRIC_DEPTH else RELATIVE_MODEL_ID

DEFAULT_YOLO_WEIGHTS = "assets/yolov8n.pt"
DEFAULT_CAM_INDEX = 0
PROC_SHORT_SIDE = 512
DEVICE = "cuda"

# Horizontal field of view of the actual camera. This sets the pixel -> bearing mapping
# that drives stereo panning, so it must match the lens: a typical laptop webcam is
# 60-70 deg, an action cam 120+. Too wide a value collapses panning to hard left/right.
FOV_DEG = 68.0

# Ground-plane suppression (metric mode). The floor is always among the nearest surfaces
# in view, so without this every frame would report an obstacle underfoot. The floor is
# measured from each frame rather than assumed, so no camera height or tilt is needed.
GROUND_FIT_TOP = 0.45        # fit the floor over rows below this fraction of the frame
GROUND_ROW_PCTL = 75         # per-row distance percentile — obstacles sit below the floor
GROUND_TOLERANCE = 0.30      # fraction of fitted distance still counted as floor
GROUND_MIN_DROP_M = 0.45     # floor must recede by this much, else it is not a floor
GROUND_MAX_RESIDUAL = 0.35   # rows must follow the fit, else the scene is not a floor

# Metric depth is true distance (larger = farther) and must be flipped to get proximity;
# a relative checkpoint already emits inverse depth (larger = nearer), so it must not be.
INVERT_DEFAULT = METRIC_DEPTH

# YOLO
YOLO_CONF = 0.30
MAX_DETS = 30

# Depth -> proximity percentile clipping
PCTL_LOW = 5
PCTL_HIGH = 95

# Proximity reference scale.
# Percentiles are measured on the background (everything outside the locked target) and
# smoothed over time, so an obstacle that grows to fill the frame cannot rescale the map
# used to judge it. 1.0 means "as near as the nearest background"; nearer reads above 1.
REF_EMA_ALPHA = 0.12      # how fast the reference scale follows the scene
MIN_BG_FRAC = 0.25        # below this much visible background, freeze the scale
PROX_CLIP = 1.5           # headroom kept above the background scale

# Looming cue: apparent size keeps rising as an obstacle closes in, even once its
# proximity has saturated. Fraction of frame area at which size alone means "max urgency".
AREA_NEAR_MIN = 0.12
AREA_NEAR_MAX = 0.55

# Temporal response: rise quickly toward danger, fall back slowly, so a single bad
# frame cannot drop a genuinely close obstacle to "far".
URGENCY_RISE = 0.55
URGENCY_FALL = 0.12

# Coasting: fade the last cue across a brief detection gap instead of cutting to silence.
COAST_DECAY = 0.82
COAST_MIN = 0.12

# Border handling
BORDER_IGNORE_FRAC = 0.10
BORDER_PENALTY_FRAC = 0.12

# Bottom-row bias (tables, low obstacles).
# Kept mild: with correctly oriented depth the floor itself is genuinely near, so a
# strong bottom bias would pull every cue toward the ground plane.
BOTTOM_GAMMA = 2.2
BOTTOM_MIN_MULT = 1.0
BOTTOM_MAX_MULT = 1.4

# --- Metric mode: thresholds are real distances -----------------------------------
ALARM_DISTANCE_M = 2.5          # start warning once an obstacle is nearer than this
RELEASE_DISTANCE_M = 3.0        # ...and keep warning until it recedes past this
CRITICAL_DISTANCE_M = 0.6       # maximum urgency at or inside this distance

# Proximity ramps from RELEASE_DISTANCE_M down to CRITICAL_DISTANCE_M, so the alarm
# distance is a positive value on that ramp. Acquiring a target requires reaching it;
# holding one only requires staying on the ramp. That gap is the hysteresis which stops
# an obstacle hovering near the threshold from flickering in and out of alarm.
MIN_TARGET_PROX_METRIC = (RELEASE_DISTANCE_M - ALARM_DISTANCE_M) / (
    RELEASE_DISTANCE_M - CRITICAL_DISTANCE_M
)
RETAIN_TARGET_PROX_METRIC = 0.02
FALLBACK_MIN_PROX_METRIC = 0.35   # an unlabelled blob must be within ~2.2 m

# --- Relative mode: anchors on the scene-relative scale ---------------------------
# 1.0 = as near as the ground at your feet; distant scenery sits near 0.30 and a
# person a few metres ahead reads 0.45-0.65.
PROX_FAR_ANCHOR = 0.30
PROX_NEAR_ANCHOR = 0.95
MIN_TARGET_PROX_RELATIVE = 0.40
RETAIN_TARGET_PROX_RELATIVE = 0.28
FALLBACK_MIN_PROX_RELATIVE = 0.80

# Relative mode only: apparent size escalates urgency once depth already says "fairly
# near", covering the range where a looming obstacle has saturated the relative scale.
# Metric mode needs no such crutch — it knows the distance outright.
SIZE_FUSION_MIN = 0.45

FLOOR_IGNORE_FRAC = 0.18  # ignore the ground immediately in front of the camera

# Target locking
IOU_LOCK_THRES = 0.20
MAX_LOST_FRAMES = 8

# Depth blob fallback (used when YOLO finds nothing)
MIN_AREA_FRAC = 0.020
MORPH_KERNEL = 9

# Audio output parameters
MIN_INTERVAL = 0.07
MAX_INTERVAL = 0.30
MIN_VOL = 0.05
MAX_VOL = 0.95
PITCH_FAR = 1200.0
PITCH_NEAR = 1200.0
