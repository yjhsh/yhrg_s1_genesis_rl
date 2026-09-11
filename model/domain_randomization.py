"""
Declarative domain randomization for scene objects (Model layer).

Motivation
----------
Object randomization used to be hard-coded inside ``PushEnv`` (``__init__`` for
size / base friction, ``_reset_idx`` for mass / friction ratio).  Adding a new
randomized quantity meant touching several places in the environment and
duplicating the "bool mask -> full batch tensor" boilerplate that Genesis'
zero-copy fast path requires.

This module turns every randomized quantity into a self-contained
:class:`IDomainRandomizer` and exposes two decorators:

  ``@randomizer(name, phase)``  (class decorator)
      Registers a randomizer class in :data:`RANDOMIZER_REGISTRY`, so it can be
      instantiated purely from configuration::

          @dataclass
          @randomizer("mass", Phase.RESET)
          class MassRandomizer(IDomainRandomizer):
              low: float = 0.05
              high: float = 0.4

  ``@randomize(phase)``  (method decorator)
      Marks an environment lifecycle method as a randomization hook.  All
      randomizers registered for that phase run *before* the method body::

          @randomize(Phase.RESET)
          def _reset_idx(self, envs_idx): ...

Phases
------
:class:`Phase.SPAWN` runs once before ``scene.build()`` -- the entity does not
exist yet, so spawn randomizers only publish a *spec* into
``RandomizationContext.state`` (e.g. the per-env morph sizes) which the spawn
method consumes.  :class:`Phase.RESET` runs on every episode reset and writes
directly into the simulator.

Configuration (single source of truth)
--------------------------------------
:class:`DomainRandomizationConfig` is a plain ``{name: kwargs}`` mapping, so
randomization is switched on/off and re-ranged without touching any code::

    cfg = DomainRandomizationConfig.from_dict({
        "size":     {"enabled": True, "low": 0.8, "high": 2.5},
        "mass":     {"enabled": True, "low": 0.05, "high": 0.4},
        "friction": {"enabled": True, "low": 0.3, "high": 1.0, "base": 0.5},
    })

Every applied value is cached per-env in ``ctx.state`` (full-batch tensors),
which doubles as the query interface for logging / observations.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import torch

from model.registry import Registry
from model.robot_config import CUBE_SIZE


__all__ = [
    "Phase",
    "RandomizationContext",
    "IDomainRandomizer",
    "RANDOMIZER_REGISTRY",
    "randomizer",
    "randomize",
    "SizeRandomizer",
    "MassRandomizer",
    "FrictionRandomizer",
    "DomainRandomizationConfig",
    "DomainRandomizerManager",
    "get_rl_push_dr_config",
]


# ─────────────────────────── registry (reuses model.registry) ───────────────────────────

# ``IDomainRandomizer`` is defined below, so the type parameter is only annotated.
RANDOMIZER_REGISTRY: "Registry[Type[IDomainRandomizer]]" = Registry("domain_randomizers")


class Phase(Enum):
    """When a randomizer runs.

    SPAWN: once, before ``scene.build()`` -- affects geometry, constant for the
           whole training run (Genesis cannot resize geoms at runtime).
    RESET: every episode reset -- affects physical quantities only.
    """

    SPAWN = "spawn"
    RESET = "reset"


# ─────────────────────────── context ───────────────────────────

@dataclass
class RandomizationContext:
    """Everything a randomizer needs. Built by the context factory of
    :func:`randomize` (see :func:`default_context_factory`).

    Attributes:
        entity: the Genesis entity being randomized (``None`` during SPAWN).
        phase: the phase this context belongs to.
        envs_idx: bool mask or index tensor selecting the envs to randomize.
        n_envs: total number of parallel environments.
        num_reset: number of selected envs (``envs_idx`` length / True count).
        device: torch device.
        scene_profile: the Model-layer scene profile (for derived quantities).
        state: per-env cache shared across phases, owned by the environment.
    """

    entity: Optional[Any]
    phase: Phase
    envs_idx: Any
    n_envs: int
    num_reset: int
    device: torch.device
    scene_profile: Any = None
    state: Dict[str, Any] = field(default_factory=dict)

    def to_batch(self, values: torch.Tensor) -> torch.Tensor:
        """Expand ``[num_reset, *]`` to ``[n_envs, *]``.

        Only for setters with Genesis' bool-mask zero-copy fast path
        (``set_pos`` / ``set_quat`` / ``set_qpos``), which expect a full-batch
        tensor.  Setters that go through ``_sanitize_io_variables``
        (``set_links_inertial_mass``, ``set_friction_ratio``, ...) must be given
        the *subset* tensor instead, because they convert a bool mask into
        indices and expect dim 0 to match ``len(envs_idx)``.
        """
        if self.envs_idx.dtype != torch.bool:
            return values
        full = torch.zeros(
            self.n_envs, *values.shape[1:], dtype=values.dtype, device=self.device
        )
        full[self.envs_idx] = values
        return full

    def store(self, key: str, values: torch.Tensor) -> None:
        """Cache per-env sampled values (full batch) into ``self.state``."""
        cache = self.state.get(key)
        if cache is None:
            cache = torch.zeros(
                self.n_envs, *values.shape[1:], dtype=values.dtype, device=self.device
            )
            self.state[key] = cache
        cache[self.envs_idx] = values


# ─────────────────────────── randomizer interface ───────────────────────────

class IDomainRandomizer(ABC):
    """Base class for a single randomized quantity.

    Subclasses are dataclasses: their fields *are* the configuration surface
    (``DomainRandomizationConfig`` passes them as kwargs).
    """

    name: str = ""
    phase: Phase = Phase.RESET

    def on_build(self, entity: Any) -> None:
        """Optional one-time setup right after ``scene.build()``.

        Used for quantities that need a base value before per-env ratios can be
        applied (e.g. base friction).
        """

    @abstractmethod
    def apply(self, ctx: RandomizationContext) -> None:
        """Sample the quantity for ``ctx.num_reset`` envs, write it into the
        simulator and cache the absolute values via ``ctx.store()``."""


def randomizer(name: str, phase: Phase) -> Callable[[Type[IDomainRandomizer]], Type[IDomainRandomizer]]:
    """Class decorator: register a randomizer under ``name`` for ``phase``."""

    def _decorate(cls: Type[IDomainRandomizer]) -> Type[IDomainRandomizer]:
        cls.name = name
        cls.phase = phase
        RANDOMIZER_REGISTRY.register(name, cls)
        return cls

    return _decorate


# ─────────────────────────── concrete randomizers ───────────────────────────

@dataclass
@randomizer("size", Phase.SPAWN)
class SizeRandomizer(IDomainRandomizer):
    """Object size randomization.

    Genesis cannot resize geoms after ``scene.build()``, so sizes are sampled
    once and consumed by the spawn method as a per-env morph list
    (heterogeneous entity). Publishes into the context state:

        ``size``        : [n_envs, 3] absolute sizes
        ``morph_sizes`` : List[Tuple[float, float, float]] for ``gs.morphs.Box``
        ``obj_z``       : spawn Z guaranteeing no table penetration
    """

    low: float = 0.8
    high: float = 2.5
    base_size: Tuple[float, float, float] = CUBE_SIZE

    def apply(self, ctx: RandomizationContext) -> None:
        scales = torch.zeros(ctx.n_envs, device=ctx.device).uniform_(self.low, self.high)
        base = torch.tensor(self.base_size, dtype=gs_float(), device=ctx.device)
        sizes = scales.unsqueeze(-1) * base

        ctx.state["size"] = sizes
        ctx.state["morph_sizes"] = [tuple(float(v) for v in s) for s in sizes.tolist()]
        ctx.state["obj_z"] = _spawn_z(ctx, float(sizes[:, 2].max().item()) / 2.0)


@dataclass
@randomizer("mass", Phase.RESET)
class MassRandomizer(IDomainRandomizer):
    """Link inertial mass randomization (uniform, in kg)."""

    low: float = 0.05
    high: float = 0.4

    def apply(self, ctx: RandomizationContext) -> None:
        mass = torch.zeros(ctx.num_reset, device=ctx.device).uniform_(self.low, self.high)
        # Subset shape [num_reset, 1]: unlike set_pos / set_quat, this setter has
        # no bool-mask zero-copy fast path, so Genesis expects dim 0 to match
        # len(envs_idx) (the *selected* envs), not n_envs.
        ctx.entity.set_links_inertial_mass(
            mass.unsqueeze(-1), links_idx_local=0, envs_idx=ctx.envs_idx
        )
        ctx.store("mass", mass)


@dataclass
@randomizer("friction", Phase.RESET)
class FrictionRandomizer(IDomainRandomizer):
    """Friction randomization.

    Genesis applies friction as ``base_friction * ratio`` with the constraint
    ``0.01 <= actual <= 5.0``, so the base value is set once after build and
    the ratio is re-sampled per episode.
    """

    low: float = 0.3
    high: float = 1.0
    base: float = 0.5

    def on_build(self, entity: Any) -> None:
        entity.set_friction(self.base)

    def apply(self, ctx: RandomizationContext) -> None:
        friction = torch.zeros(ctx.num_reset, device=ctx.device).uniform_(self.low, self.high)
        ratio = (friction / self.base).clamp(0.01 / self.base, 5.0 / self.base)
        # Subset shape [num_reset, 1] -- see MassRandomizer.apply.
        ctx.entity.set_friction_ratio(
            ratio.unsqueeze(-1), links_idx_local=0, envs_idx=ctx.envs_idx
        )
        ctx.store("friction", friction)


@functools.lru_cache(maxsize=1)
def gs_float() -> torch.dtype:
    """Genesis float dtype.

    Resolved lazily so this module stays importable without an initialized
    simulator (and testable on CPU).
    """
    try:
        import genesis as gs

        return gs.tc_float
    except Exception:  # pragma: no cover - genesis not initialized
        return torch.float32


def _spawn_z(ctx: RandomizationContext, max_half_z: float) -> float:
    """Object center Z that keeps every size variant above the table top."""
    sp = ctx.scene_profile
    return sp.table_top_z + max_half_z + sp.z_eps


# ─────────────────────────── configuration & manager ───────────────────────────

@dataclass
class DomainRandomizationConfig:
    """Declarative randomization configuration: ``{name: kwargs}``.

    ``enabled=False`` (or simply omitting the entry) switches a randomizer off.
    All other kwargs are forwarded to the registered randomizer's constructor,
    i.e. they are exactly its dataclass fields.
    """

    specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, specs: Dict[str, Dict[str, Any]]) -> "DomainRandomizationConfig":
        return cls(specs={name: dict(kwargs) for name, kwargs in specs.items()})

    @classmethod
    def from_yaml(cls, path: str) -> "DomainRandomizationConfig":
        import yaml  # lazy: only needed when configuring from a YAML file

        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: dict(kwargs) for name, kwargs in self.specs.items()}


class DomainRandomizerManager:
    """Instantiates randomizers from config and applies them by phase."""

    def __init__(self, config: DomainRandomizationConfig) -> None:
        self.config = config
        self.randomizers: List[IDomainRandomizer] = []
        for name, kwargs in config.specs.items():
            kwargs = dict(kwargs)
            if not kwargs.pop("enabled", True):
                continue
            self.randomizers.append(RANDOMIZER_REGISTRY.create(name, **kwargs))

    @property
    def names(self) -> List[str]:
        return [r.name for r in self.randomizers]

    def prepare(self, entity: Any) -> None:
        """One-time post-build setup for every enabled randomizer."""
        for r in self.randomizers:
            r.on_build(entity)

    def apply(self, phase: Phase, ctx: RandomizationContext) -> None:
        """Run every enabled randomizer belonging to ``phase``."""
        for r in self.randomizers:
            if r.phase is phase:
                r.apply(ctx)


def get_rl_push_dr_config() -> DomainRandomizationConfig:
    """Default randomization config for the RL pushing task.

    Randomization is opt-in: override per experiment via
    ``PushEnv(dr_config=DomainRandomizationConfig.from_dict({...}))``.
    """
    return DomainRandomizationConfig.from_dict(
        {
            "size": {"enabled": False, "low": 0.8, "high": 2.5},
            "mass": {"enabled": False, "low": 0.05, "high": 0.4},
            "friction": {"enabled": False, "low": 0.3, "high": 1.0, "base": 0.5},
        }
    )


# ─────────────────────────── method decorator ───────────────────────────

def default_context_factory(env: Any, phase: Phase, args: Sequence[Any], kwargs: Dict[str, Any]) -> RandomizationContext:
    """Build a context from the conventional attributes of the environment.

    SPAWN randomizes all envs; RESET uses the first positional argument
    (``envs_idx``) of the decorated method.
    """
    if phase is Phase.SPAWN:
        envs_idx = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        num_reset = env.num_envs
    else:
        envs_idx = args[0] if args else kwargs["envs_idx"]
        num_reset = (
            int(envs_idx.sum().item())
            if envs_idx.dtype == torch.bool
            else len(envs_idx)
        )
    return RandomizationContext(
        entity=getattr(env, "dr_entity", None),
        phase=phase,
        envs_idx=envs_idx,
        n_envs=env.num_envs,
        num_reset=num_reset,
        device=env.device,
        scene_profile=getattr(env, "scene_profile", None),
    )


def randomize(
    phase: Phase,
    ctx_factory: Optional[Callable[..., RandomizationContext]] = None,
) -> Callable:
    """Method decorator: apply all randomizers of ``phase`` before the method.

    Args:
        phase: :class:`Phase.SPAWN` or :class:`Phase.RESET`.
        ctx_factory: optional ``(env, phase, args, kwargs) -> RandomizationContext``
            override for environments that do not follow the default conventions.

    The decorated method's owner must provide:

        ``dr_manager``: :class:`DomainRandomizerManager`
        ``dr_state``   : dict holding the per-env cache (shared across phases)
        ``dr_entity``  : the entity being randomized (``None`` during SPAWN)
    """

    factory = ctx_factory or default_context_factory

    def _decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            manager = getattr(self, "dr_manager", None)
            if manager is not None:
                ctx = factory(self, phase, args, kwargs)
                # Guard: lifecycle methods usually return early when no env is
                # selected, and Genesis rejects set_* calls with an empty idx.
                if ctx.num_reset > 0:
                    ctx.state = self.dr_state
                    manager.apply(phase, ctx)
            return fn(self, *args, **kwargs)

        wrapper.__dr_phase__ = phase  # introspection / tests
        return wrapper

    return _decorate
