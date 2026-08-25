"""
Unified Registry base class.

Encapsulates the common logic for type registration and lookup, eliminating
duplicated Registry implementations across controller/modes.py, view/proxy.py,
model/factories.py and model/loaders.py.

Core functionality:
  1. register(name, cls): register a type
  2. create(name, *args, **kwargs): look up and instantiate
  3. get_available(): get all registered names
"""

import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic type registry.

    Type Parameter:
        T: the base class type being registered.

    Usage:
        registry = Registry[RobotConfigFactory]("robot_factories")
        registry.register("yhrg_s1", YHRGS1RobotConfigFactory)
        factory = registry.create("yhrg_s1", urdf_path="/path/to/urdf")
    """

    def __init__(self, name: str = "Registry"):
        """
        Args:
            name: registry name, used for logging and error messages.
        """
        self._name = name
        self._entries: Dict[str, Type[T]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: str, entry_class: Type[T]) -> "Registry[T]":
        """
        Register a type.

        Args:
            name: type identifier.
            entry_class: the class to register.

        Returns:
            self, to support method chaining.
        """
        if name in self._entries:
            logger.warning(f"[{self._name}] Overwriting existing entry: {name}")
        self._entries[name] = entry_class
        logger.info(f"[{self._name}] Registered: {name} -> {entry_class.__name__}")
        return self

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Look up and instantiate a registered type.

        Args:
            name: type identifier.
            *args, **kwargs: arguments passed to the constructor.

        Returns:
            the instantiated object.

        Raises:
            ValueError: if the name is not registered.
        """
        if name not in self._entries:
            raise ValueError(
                f"[{self._name}] Unknown entry: '{name}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[name](*args, **kwargs)

    def get(self, name: str) -> Type[T]:
        """
        Get the registered class (without instantiating it).

        Args:
            name: type identifier.

        Returns:
            the registered class.

        Raises:
            ValueError: if the name is not registered.
        """
        if name not in self._entries:
            raise ValueError(
                f"[{self._name}] Unknown entry: '{name}'. "
                f"Available: {list(self._entries.keys())}"
            )
        return self._entries[name]

    def get_available(self) -> List[str]:
        """获取所有已注册的名称列表。"""
        return list(self._entries.keys())

    def has(self, name: str) -> bool:
        """检查名称是否已注册。"""
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"{self._name}({list(self._entries.keys())})"
