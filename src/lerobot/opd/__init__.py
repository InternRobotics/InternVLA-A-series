"""On-policy distillation utilities for continuous-action VLA policies."""

from lerobot.opd.config import OPDConfig

OPD_IMAGE_PREFIX = "opd.image"
OPD_IMAGE_MASK = "opd.image_mask"
OPD_IS_ON_POLICY = "opd.is_on_policy"
OPD_RAW_STATE = "opd.raw_state"
OPD_ROBOT_TYPE = "opd.robot_type"
OPD_TASK = "opd.task"
OPD_TEACHER_ACTION_MASK = "opd.teacher_action_mask"
OPD_TEACHER_ACTION_MEAN = "opd.teacher_action_mean"
OPD_TEACHER_ACTION_STD = "opd.teacher_action_std"

__all__ = [
    "OPDConfig",
    "OPD_IMAGE_MASK",
    "OPD_IMAGE_PREFIX",
    "OPD_IS_ON_POLICY",
    "OPD_RAW_STATE",
    "OPD_ROBOT_TYPE",
    "OPD_TASK",
    "OPD_TEACHER_ACTION_MASK",
    "OPD_TEACHER_ACTION_MEAN",
    "OPD_TEACHER_ACTION_STD",
]
