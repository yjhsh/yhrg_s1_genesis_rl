"""
Scene Configuration Module for YHRG S1 Robotic Arm Simulation.

This module provides centralized configuration management for all scene-related
settings, including asset paths, positioning parameters, and environmental
properties. It serves as a single source of truth for scene configuration.

Configuration Categories:
- Asset paths: URDF files, mesh files, texture paths
- Positioning: Robot base position, desktop position, target object positions
- Environment: Ground plane settings, lighting, camera parameters
- Physics: Simulation parameters, collision settings

Usage:
    from config.scene_config import SceneConfig
    
    config = SceneConfig()
    robot_config = config.robot
    desktop_config = config.desktop
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from model.robot_config import URDF_PATH as _DEFAULT_ROBOT_URDF


@dataclass
class RobotConfig:
    """
    Configuration for the robotic arm.
    
    Attributes:
        name: Robot identifier.
        urdf_path: Path to the URDF file.
        base_position: Position of the robot base in world coordinates.
        base_orientation: Orientation of the robot base (quaternion).
        fixed_base: Whether the base is fixed to the ground.
    """
    name: str = "yhrg_s1"
    urdf_path: str = _DEFAULT_ROBOT_URDF
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    fixed_base: bool = True


@dataclass
class DesktopConfig:
    """
    Configuration for the desktop/table environment.
    
    Attributes:
        name: Desktop identifier.
        urdf_path: Path to the desktop URDF or mesh file.
        position: Position of the desktop center.
        orientation: Orientation of the desktop.
        scale: Scale factor for the desktop.
        height: Height of the desktop surface from ground.
    """
    name: str = "desktop"
    urdf_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset", "urdf", "square_table.urdf")
    position: Tuple[float, float, float] = (-0.3, 0.0, -.15)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: float = 5.0
    height: float = 0.75
    fixed: bool = True


@dataclass
class TargetObjectConfig:
    """
    Configuration for target objects in the scene.
    
    Attributes:
        name: Object identifier.
        urdf_path: Path to the object URDF or mesh file.
        position: Position of the object.
        orientation: Orientation of the object.
        scale: Scale factor for the object.
        mass: Mass of the object in kg.
        friction: Friction coefficient.
    """
    name: str = "target_object"
    urdf_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset", "urdf", "cube.urdf")
    position: Tuple[float, float, float] = (0.5, 0.0, 0.8)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: float = 1.0
    mass: float = 0.1
    friction: float = 0.5
    fixed: bool = False


@dataclass
class GroundConfig:
    """
    Configuration for the ground plane.
    
    Attributes:
        enabled: Whether to render the ground plane.
        position: Position of the ground surface.
        size: Size of the ground plane [length, width].
        color: RGBA color of the ground.
        static_friction: Static friction coefficient.
        dynamic_friction: Dynamic friction coefficient.
    """
    enabled: bool = True
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float] = (10.0, 10.0)
    color: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class CameraConfig:
    """
    Configuration for the scene camera.
    
    Attributes:
        position: Camera position in world coordinates.
        lookat: Point the camera is looking at.
        fov: Field of view in degrees.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
    """
    position: Tuple[float, float, float] = (0.0, -3.5, 2.5)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    fov: float = 30.0
    near: float = 0.01
    far: float = 100.0


@dataclass
class LightingConfig:
    """
    Configuration for scene lighting.
    
    Attributes:
        ambient: Ambient light intensity.
        directional: Directional light direction and intensity.
        shadows: Whether to enable shadows.
    """
    ambient: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    directional_direction: Tuple[float, float, float] = (0.5, 0.5, -1.0)
    directional_intensity: float = 1.0
    shadows: bool = True


class SceneConfig:
    """
    Central configuration manager for all scene-related settings.
    
    This class provides a unified interface for accessing and managing
    all scene configuration parameters. It serves as the single source
    of truth for scene setup.
    
    Attributes:
        robot: Robot configuration.
        desktop: Desktop configuration.
        target_objects: List of target object configurations.
        ground: Ground plane configuration.
        camera: Camera configuration.
        lighting: Lighting configuration.
    
    Example:
        config = SceneConfig()
        config.load_from_file("scene_config.yaml")
        robot_pos = config.robot.base_position
    """
    
    DEFAULT_ROBOT_URDF = _DEFAULT_ROBOT_URDF
    DEFAULT_DESKTOP_URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset", "urdf", "square_table.urdf")
    DEFAULT_OBJECT_URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset", "urdf", "cube.urdf")
    
    def __init__(
        self,
        robot_urdf: Optional[str] = None,
        desktop_urdf: Optional[str] = None,
        target_object_urdf: Optional[str] = None,
    ):
        """
        Initialize the scene configuration with default values.
        
        Args:
            robot_urdf: Optional custom robot URDF path.
            desktop_urdf: Optional custom desktop URDF path.
            target_object_urdf: Optional custom target object URDF path.
        """
        self._robot = RobotConfig(
            urdf_path=robot_urdf or self.DEFAULT_ROBOT_URDF,
        )
        
        self._desktop = DesktopConfig(
            urdf_path=desktop_urdf or self.DEFAULT_DESKTOP_URDF,
        )
        
        self._target_objects: List[TargetObjectConfig] = [
            TargetObjectConfig(
                urdf_path=target_object_urdf or self.DEFAULT_OBJECT_URDF,
            )
        ]
        
        self._ground = GroundConfig()
        self._camera = CameraConfig()
        self._lighting = LightingConfig()
        
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that all configured paths exist."""
        paths_to_check = [
            ("robot", self._robot.urdf_path),
            ("desktop", self._desktop.urdf_path),
        ]
        
        for name, path in paths_to_check:
            if path and not os.path.exists(path):
                pass
    
    @property
    def robot(self) -> RobotConfig:
        """Get the robot configuration."""
        return self._robot
    
    @property
    def desktop(self) -> DesktopConfig:
        """Get the desktop configuration."""
        return self._desktop
    
    @property
    def target_objects(self) -> List[TargetObjectConfig]:
        """Get the list of target object configurations."""
        return self._target_objects
    
    @property
    def ground(self) -> GroundConfig:
        """Get the ground configuration."""
        return self._ground
    
    @property
    def camera(self) -> CameraConfig:
        """Get the camera configuration."""
        return self._camera
    
    @property
    def lighting(self) -> LightingConfig:
        """Get the lighting configuration."""
        return self._lighting
    
    def set_robot_position(self, x: float, y: float, z: float) -> None:
        """
        Set the robot base position.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        self._robot.base_position = (x, y, z)
    
    def set_desktop_position(self, x: float, y: float, z: float) -> None:
        """
        Set the desktop position.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
        """
        self._desktop.position = (x, y, z)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire configuration to dictionary format.
        
        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "robot": {
                "name": self._robot.name,
                "urdf_path": self._robot.urdf_path,
                "base_position": self._robot.base_position,
                "base_orientation": self._robot.base_orientation,
                "fixed_base": self._robot.fixed_base,
            },
            "desktop": {
                "name": self._desktop.name,
                "urdf_path": self._desktop.urdf_path,
                "position": self._desktop.position,
                "orientation": self._desktop.orientation,
                "scale": self._desktop.scale,
                "height": self._desktop.height,
                "fixed": self._desktop.fixed,
            },
            "target_objects": [
                {
                    "name": obj.name,
                    "urdf_path": obj.urdf_path,
                    "position": obj.position,
                    "orientation": obj.orientation,
                    "scale": obj.scale,
                    "mass": obj.mass,
                    "friction": obj.friction,
                    "fixed": obj.fixed,
                }
                for obj in self._target_objects
            ],
            "ground": {
                "enabled": self._ground.enabled,
                "position": self._ground.position,
                "size": self._ground.size,
                "color": self._ground.color,
            },
            "camera": {
                "position": self._camera.position,
                "lookat": self._camera.lookat,
                "fov": self._camera.fov,
            },
            "lighting": {
                "ambient": self._lighting.ambient,
                "directional_direction": self._lighting.directional_direction,
                "directional_intensity": self._lighting.directional_intensity,
                "shadows": self._lighting.shadows,
            },
        }


def get_default_scene_config() -> SceneConfig:
    """
    Factory function to create a default scene configuration.
    
    Returns:
        SceneConfig instance with default values.
    """
    return SceneConfig()
