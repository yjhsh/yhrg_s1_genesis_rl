"""
Robot Configuration Factories for Model Layer.

This module provides factory classes for creating robot configurations
using the Factory Pattern. These factories encapsulate the creation logic
for different robot models.

All robot constants are imported from model.robot_config (Single Source of Truth).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np

from model.robot_config import (
    URDF_PATH,
    JOINT_NAMES,
    TOTAL_DOF,
    BASE_LINK_NAME,
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
    CONTROL_KP,
    CONTROL_KV,
    FORCE_LIMITS_LOWER,
    FORCE_LIMITS_UPPER,
    NEUTRAL_POSITION,
    HOME_POSITION,
    RL_PUSH_DEFAULT_POSITION,
    get_full_config,
)
from model.registry import Registry

logger = logging.getLogger(__name__)


class RobotConfigFactory(ABC):
    """
    Abstract factory for creating robot configurations.
    
    Defines the contract for creating robot configuration dictionaries
    that can be used throughout the simulation.
    """
    
    @abstractmethod
    def create_config(self) -> Dict[str, Any]:
        """
        Create and return the robot configuration dictionary.
        
        Returns:
            Dictionary containing robot configuration parameters.
        """
        pass
    
    @abstractmethod
    def get_robot_name(self) -> str:
        """
        Return the robot name identifier.
        
        Returns:
            String identifier for the robot.
        """
        pass
    
    @abstractmethod
    def get_joint_names(self) -> Tuple[str, ...]:
        """
        Return the ordered tuple of joint names.
        
        Returns:
            Tuple of joint name strings.
        """
        pass
    
    @abstractmethod
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the joint position limits.
        
        Returns:
            Tuple of (lower_limits, upper_limits).
        """
        pass
    
    @abstractmethod
    def get_control_gains(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the PD control gains.
        
        Returns:
            Tuple of (kp, kv) gain arrays.
        """
        pass
    
    @abstractmethod
    def get_force_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the force limits for each joint.
        
        Returns:
            Tuple of (lower_force, upper_force).
        """
        pass


class YHRGS1RobotConfigFactory(RobotConfigFactory):
    """
    Factory for creating YHRG S1 robotic arm configuration.
    
    All constants are imported from model.robot_config (Single Source of Truth).
    The robot has:
    - 6 revolute joints for the arm (1joint through 6joint)
    - 2 prismatic joints for the gripper fingers (7joint, 8joint)
    - Fixed base link for stability
    """
    
    def __init__(self, urdf_path: str = ""):
        """
        Initialize the YHRG S1 robot configuration factory.
        
        Args:
            urdf_path: Absolute path to the URDF file. Defaults to model.robot_config.URDF_PATH.
        """
        self._urdf_path = urdf_path or URDF_PATH
        logger.info(f"YHRGS1RobotConfigFactory created for: {self._urdf_path}")
    
    def create_config(self) -> Dict[str, Any]:
        """
        Create and return the complete robot configuration dictionary.
        
        Returns:
            Dictionary containing all robot configuration parameters.
        """
        return get_full_config(self._urdf_path)
    
    def get_robot_name(self) -> str:
        return "yhrg_s1"
    
    def get_joint_names(self) -> Tuple[str, ...]:
        return JOINT_NAMES
    
    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return (JOINT_LIMITS_LOWER.copy(), JOINT_LIMITS_UPPER.copy())
    
    def get_control_gains(self) -> Tuple[np.ndarray, np.ndarray]:
        return (CONTROL_KP.copy(), CONTROL_KV.copy())
    
    def get_force_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return (FORCE_LIMITS_LOWER.copy(), FORCE_LIMITS_UPPER.copy())
    
    def get_urdf_path(self) -> str:
        """Return the URDF file path."""
        return self._urdf_path
    
    def is_fixed_base(self) -> bool:
        """Return whether the robot has a fixed base."""
        return True
    
    def get_base_link_name(self) -> str:
        """Return the name of the base link."""
        return BASE_LINK_NAME
    
    def get_total_dof(self) -> int:
        """Return the total degrees of freedom."""
        return TOTAL_DOF
    
    def get_default_positions(self) -> Dict[str, np.ndarray]:
        """
        Return the default positions.
        
        Returns:
            Dictionary with 'neutral', 'home', and 'rl_push' positions.
        """
        return {
            "neutral": NEUTRAL_POSITION.copy(),
            "home": HOME_POSITION.copy(),
            "rl_push": RL_PUSH_DEFAULT_POSITION.copy(),
        }


# Module-level registry for robot config factories
_factory_registry = Registry[RobotConfigFactory]("robot_config_factories")
_factory_registry.register("yhrg_s1", YHRGS1RobotConfigFactory)


class RobotConfigFactoryRegistry:
    """
    Registry for robot configuration factories, backed by model.registry.Registry.
    
    Enables dynamic factory lookup and creation by robot name,
    supporting extensibility for new robot models.
    """
    
    @classmethod
    def register(cls, name: str, factory_class: Type[RobotConfigFactory]) -> None:
        _factory_registry.register(name, factory_class)
    
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> RobotConfigFactory:
        return _factory_registry.create(name, *args, **kwargs)
    
    @classmethod
    def get_available_robots(cls) -> list:
        return _factory_registry.get_available()
