DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEFAULT_CAM_INDEX = 0
PROC_SHORT_SIDE = 512
DEVICE = "cuda"

# Game loop timing (seconds)
CUE_DURATION      = 1.5   # how long the zone tone plays
RESPONSE_WINDOW   = 3.0   # how long the patient has to orient
FEEDBACK_DURATION = 0.8   # how long the correct/wrong overlay shows

# Level progression: unlock next level after this accuracy over last N rounds
UNLOCK_ACCURACY = 0.70
UNLOCK_ROUNDS   = 10

# Zone definitions per level.
# Each zone has: yaw threshold boundaries, stereo pan, pitch (Hz), and display label.
# Zones are ordered left → right.  Yaw: positive = looking right, negative = left.
ZONE_CONFIGS = {
    1: {  # 3 zones
        "n": 3,
        "yaw_bins": [-20.0, 20.0],          # thresholds between zones
        "pans":     [-0.9,  0.0,  0.9],
        "pitches":  [440.0, 523.0, 659.0],  # A4, C5, E5 — a major triad
        "labels":   ["Left", "Center", "Right"],
    },
    2: {  # 5 zones
        "n": 5,
        "yaw_bins": [-35.0, -12.0, 12.0, 35.0],
        "pans":     [-1.0,  -0.5,  0.0,  0.5,  1.0],
        "pitches":  [349.0, 440.0, 523.0, 659.0, 784.0],  # F4, A4, C5, E5, G5
        "labels":   ["Far Left", "Left", "Center", "Right", "Far Right"],
    },
    3: {  # 9 zones (3 columns × 3 rows — uses yaw + pitch)
        "n": 9,
        "yaw_bins":   [-20.0, 20.0],
        "pitch_bins": [10.0, -10.0],   # > 10° = top row, < -10° = bottom row
        # Index = col * 3 + row (col 0=left, row 0=top)
        "pans":    [-0.9, -0.9, -0.9,  0.0,  0.0,  0.0,  0.9,  0.9,  0.9],
        "pitches": [880.0, 440.0, 220.0, 1047.0, 523.0, 262.0, 1175.0, 659.0, 330.0],
        "labels":  ["TL", "ML", "BL", "TC", "MC", "BC", "TR", "MR", "BR"],
    },
}

