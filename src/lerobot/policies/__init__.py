"""Policy configs exposed without importing every model dependency eagerly."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .g05.configuration_g05 import G05Config
    from .internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
    from .lingbot_vla_2.configuration_lingbot_vla_2 import LingBotVLA2Config
    from .pi0.configuration_pi0 import PI0Config
    from .pi0_fast.configuration_pi0_fast import PI0FastConfig
    from .pi05.configuration_pi05 import PI05Config

__all__ = [
    "G05Config",
    "InternVLAA15Config",
    "LingBotVLA2Config",
    "PI0Config",
    "PI0FastConfig",
    "PI05Config",
]


def __getattr__(name: str):
    """Load only the selected policy's config and its versioned dependencies."""
    if name == "G05Config":
        from .g05.configuration_g05 import G05Config

        return G05Config
    if name == "InternVLAA15Config":
        from .internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config

        return InternVLAA15Config
    if name == "LingBotVLA2Config":
        from .lingbot_vla_2.configuration_lingbot_vla_2 import LingBotVLA2Config

        return LingBotVLA2Config
    if name == "PI0Config":
        from .pi0.configuration_pi0 import PI0Config

        return PI0Config
    if name == "PI0FastConfig":
        from .pi0_fast.configuration_pi0_fast import PI0FastConfig

        return PI0FastConfig
    if name == "PI05Config":
        from .pi05.configuration_pi05 import PI05Config

        return PI05Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
