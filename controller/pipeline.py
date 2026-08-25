"""
统一Pipeline基类。

封装Handler链式调用（Chain of Responsibility）的通用逻辑，
消除 controller/handlers.py、view/flow.py 中的Pipeline重复实现。

核心流程:
  1. add_handler(): 添加Handler并自动链接前驱节点
  2. execute(): 从第一个Handler开始执行链式调用
  3. step(): 执行单个步骤（可指定特定Handler）
  4. set_context()/get_context(): 管理上下文数据
"""

import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Pipeline(Generic[T]):
    """
    通用Pipeline实现，支持Handler链式调用。

    Type Parameter:
        T: Handler类型，需实现 handle(context) -> context 方法。

    Usage:
        pipeline = Pipeline()
        pipeline.add_handler(handler_a).add_handler(handler_b)
        result = pipeline.execute()
    """

    def __init__(self, name: str = "Pipeline"):
        """
        Args:
            name: Pipeline名称，用于日志。
        """
        self._name = name
        self._handlers: List[T] = []
        self._context: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def handlers(self) -> List[T]:
        return self._handlers

    def add_handler(self, handler: T) -> "Pipeline[T]":
        """
        添加Handler到Pipeline末尾，自动链接前驱节点。

        Args:
            handler: 需实现 handle(context) 方法和 set_next(handler) 方法的对象。

        Returns:
            self，支持链式调用。
        """
        if self._handlers:
            prev = self._handlers[-1]
            if hasattr(prev, "set_next"):
                prev.set_next(handler)
        self._handlers.append(handler)
        logger.debug(f"[{self._name}] Added handler: {type(handler).__name__}")
        return self

    def execute(self) -> Dict[str, Any]:
        """
        从第一个Handler开始执行链式调用。

        Returns:
            最终的context字典。

        Raises:
            RuntimeError: Pipeline中没有Handler。
        """
        if not self._handlers:
            raise RuntimeError(f"[{self._name}] No handlers in pipeline")
        logger.debug(f"[{self._name}] Executing pipeline with {len(self._handlers)} handlers")
        return self._handlers[0].handle(self._context)

    def step(self) -> Dict[str, Any]:
        """
        执行Pipeline的一个步骤。默认等同于execute()。

        子类可重写此方法以实现特定逻辑（如查找特定Handler执行）。

        Returns:
            执行结果context字典。
        """
        return self.execute()

    def set_context(self, key: str, value: Any) -> "Pipeline[T]":
        """设置上下文数据。"""
        self._context[key] = value
        return self

    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文数据。"""
        return self._context.get(key, default)

    def clear_context(self) -> "Pipeline[T]":
        """清除所有上下文数据。"""
        self._context.clear()
        return self

    def __len__(self) -> int:
        return len(self._handlers)

    def __repr__(self) -> str:
        handler_names = [type(h).__name__ for h in self._handlers]
        return f"{self._name}({handler_names})"
