"""
统一键盘控制模块。

集中定义YHRG S1机械臂的键盘输入处理逻辑和终端键盘监听器，
消除 controller/flow.py、controller/modes.py、controller/handlers.py 中的重复实现。

按键映射:
  - 1-8: 选择关节
  - UP/DOWN: 调整选中关节角度（步长 joint_step）
  - LEFT/RIGHT: 调整夹爪开合（步长 gripper_step）
  - R: 重置到Home位置
  - N: 重置到Neutral位置
  - Q: 退出
"""

import logging
import sys
import termios
import threading
import tty
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from model.robot_config import (
    ARM_DOF,
    GRIPPER_DOF,
    TOTAL_DOF,
    HOME_POSITION,
    NEUTRAL_POSITION,
)

logger = logging.getLogger(__name__)

# ── 按键映射常量 ──
KEY_BINDINGS: Dict[str, str] = {
    "1": "select_joint_1",
    "2": "select_joint_2",
    "3": "select_joint_3",
    "4": "select_joint_4",
    "5": "select_joint_5",
    "6": "select_joint_6",
    "7": "select_joint_7",
    "8": "select_joint_8",
    "UP": "joint_increase",
    "DOWN": "joint_decrease",
    "LEFT": "gripper_close",
    "RIGHT": "gripper_open",
    "r": "reset_home",
    "R": "reset_home",
    "n": "reset_neutral",
    "N": "reset_neutral",
    "q": "quit",
    "Q": "quit",
}

# 终端方向键转义序列映射
ARROW_KEY_MAP: Dict[str, str] = {
    "\x1b[A": "UP",
    "\x1b[B": "DOWN",
    "\x1b[C": "RIGHT",
    "\x1b[D": "LEFT",
    "\x1bOA": "UP",      # 应用键盘模式
    "\x1bOB": "DOWN",
    "\x1bOC": "RIGHT",
    "\x1bOD": "LEFT",
}


class KeyboardInputProcessor:
    """
    键盘输入处理器：将按键映射为关节目标位置修改。

    纯逻辑类，不依赖RobotState或SimulationState。
    接收当前target数组和按键，返回修改后的target数组。

    Usage:
        processor = KeyboardInputProcessor()
        target = np.zeros(8)
        target, action = processor.process_key("1", target)
        target, action = processor.process_key("UP", target)
    """

    def __init__(
        self,
        joint_step: float = 0.05,
        gripper_step: float = 0.005,
    ):
        """
        Args:
            joint_step: 关节角度调整步长（弧度）。
            gripper_step: 夹爪开合调整步长。
        """
        self.joint_step = joint_step
        self.gripper_step = gripper_step
        self._selected_joint: int = 0  # 0-indexed, 0~7

    @property
    def selected_joint(self) -> int:
        """当前选中的关节索引（0-indexed）。"""
        return self._selected_joint

    def process_key(self, key: str, target: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        处理按键输入，修改target数组。

        Args:
            key: 按键字符串（大写，如 "UP", "DOWN", "1"~"8", "r", "n", "q"）。
            target: 当前关节目标位置数组（长度=TOTAL_DOF）。

        Returns:
            (modified_target, action_name):
                modified_target: 修改后的目标位置数组（新数组，不修改原数组）。
                action_name: 动作名称，用于日志和状态更新。
        """
        result = target.copy()
        action = "none"

        # 关节选择
        if key.isdigit() and 1 <= int(key) <= TOTAL_DOF:
            self._selected_joint = int(key) - 1
            action = f"select_joint_{key}"
            logger.debug(f"Selected joint {key} (index {self._selected_joint})")
            return result, action

        # 关节角度调整
        if key == "UP":
            result[self._selected_joint] += self.joint_step
            action = "joint_increase"
        elif key == "DOWN":
            result[self._selected_joint] -= self.joint_step
            action = "joint_decrease"
        elif key == "LEFT":
            # 夹爪闭合
            for i in range(ARM_DOF, ARM_DOF + GRIPPER_DOF):
                result[i] += self.gripper_step
            action = "gripper_close"
        elif key == "RIGHT":
            # 夹爪打开
            for i in range(ARM_DOF, ARM_DOF + GRIPPER_DOF):
                result[i] -= self.gripper_step
            action = "gripper_open"
        elif key in ("r", "R"):
            result = HOME_POSITION.copy()
            action = "reset_home"
        elif key in ("n", "N"):
            result = NEUTRAL_POSITION.copy()
            action = "reset_neutral"
        elif key in ("q", "Q"):
            action = "quit"

        return result, action

    def get_key_bindings(self) -> Dict[str, str]:
        """返回当前按键映射配置。"""
        return KEY_BINDINGS.copy()


class TerminalKeyboardListener:
    """
    终端键盘监听器：在独立线程中捕获键盘输入。

    使用termios原始模式读取按键，支持方向键转义序列解析。
    线程安全：通过Lock保护按键缓冲区。

    Usage:
        listener = TerminalKeyboardListener()
        listener.start()
        key = listener.get_key()  # 非阻塞，返回最新按键或None
        listener.stop()
    """

    def __init__(self):
        self._key: Optional[str] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动键盘监听线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info("TerminalKeyboardListener started")

    def stop(self) -> None:
        """停止键盘监听线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("TerminalKeyboardListener stopped")

    def get_key(self) -> Optional[str]:
        """
        获取最新的按键（非阻塞）。

        Returns:
            按键字符串或None（无新按键）。获取后清除缓冲区。
        """
        with self._lock:
            key = self._key
            self._key = None
            return key

    def _listen(self) -> None:
        """监听线程主循环。"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1)
                key = self._parse_key(ch)
                if key is not None:
                    with self._lock:
                        self._key = key
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _parse_key(self, ch: str) -> Optional[str]:
        """解析按键，处理方向键转义序列。"""
        if ch == "\x1b":
            # 可能是方向键转义序列
            import select
            if select.select([sys.stdin], [], [], 0.01)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        ch3 = sys.stdin.read(1)
                        seq = f"\x1b[{ch3}"
                        return ARROW_KEY_MAP.get(seq)
                elif ch2 == "O":
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        ch3 = sys.stdin.read(1)
                        seq = f"\x1bO{ch3}"
                        return ARROW_KEY_MAP.get(seq)
            return None  # 忽略未识别的转义序列
        elif ch == "\r" or ch == "\n":
            return "ENTER"
        elif ch == " ":
            return "SPACE"
        elif ch.isprintable():
            return ch
        return None
