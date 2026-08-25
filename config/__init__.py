"""
Configuration package for YHRG S1 robotic arm simulation.

This package provides centralized configuration management for all
simulation-related settings, including scene, robot, and environment
configurations.

Modules:
    scene_config: Scene configuration management
"""

from .scene_config import (
    SceneConfig,
    RobotConfig,
    DesktopConfig,
    TargetObjectConfig,
    GroundConfig,
    CameraConfig,
    LightingConfig,
    get_default_scene_config,
)

__all__ = [
    "SceneConfig",
    "RobotConfig",
    "DesktopConfig",
    "TargetObjectConfig",
    "GroundConfig",
    "CameraConfig",
    "LightingConfig",
    "get_default_scene_config",
]
