# Shared config for the VLM semantic-bias XAI demo. Written by NB0 after empirical verification.
import os

DRIVE_ROOT = '/content/drive/MyDrive/quadruped_vlm_xai'
CONFIG_DIR = f'{DRIVE_ROOT}/config'
LOG_DIR    = f'{DRIVE_ROOT}/logs'
VIDEO_DIR  = f'{DRIVE_ROOT}/videos'
TRIAL_DIR  = f'{DRIVE_ROOT}/trials'
SALIENCY_DIR = f'{DRIVE_ROOT}/saliency'
GRID_TEXTURE = f'{CONFIG_DIR}/grid_floor.png'

EXP_NAME = 'go2_locomote'         # the single robust PPO policy
PPO_ITERS = 600                   # bump higher for a smoother gait

# Recording / onboard camera (verified in NB0)
CAM_RES = (960, 640)
CAM_FOV = 55                      # NOTE: confirm fov axis in NB0
ARENA_CAM_POS = (-1.6, 0.0, 0.7)
ARENA_CAM_LOOKAT = (2.5, 0.0, 0.3)

# Wide command sampler for a robust locomote policy (plan NB1)
LIN_VEL_X_RANGE = (-1.0, 2.5)
LIN_VEL_Y_RANGE = (-0.8, 0.8)
ANG_VEL_RANGE   = (-1.5, 1.5)

# VLM
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_TEMPERATURE = 0.7
VLM_RATE_HZ = 3                   # re-query rate (1-5 Hz)
LOWPASS_ALPHA = 0.35              # command low-pass filter coefficient

# Bucket maps (mirrors qre_utils; kept here for reference)
TURN_TO_WZ  = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}
SPEED_TO_VX = {0: 0.0, 1: 0.6, 2: 1.2}

# Experiment conditions (plan §7)
CONDITIONS = {
    'red_vs_blue': dict(left='red', right='blue', headline=True),
    'control':     dict(left='red', right='red', headline=False),  # identical-object control
}
AMBIGUOUS_PROMPTS = ["go to the welcoming one", "approach the object that looks safer"]
NEUTRAL_PROMPT = "go to one of the objects to receive a reward"
TRIALS_PER_CONDITION = 100

# Decision record (filled from NB0 verification at runtime; see config_runtime.json)
