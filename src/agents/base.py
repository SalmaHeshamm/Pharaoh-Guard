"""
Common agent interface. Every concrete agent implements `run(state)` and
returns the *partial* state update (LangGraph merges it into the shared
GraphState) — agents never mutate the input state in place.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Read what it needs from `state`, return only the keys it updates."""
        raise NotImplementedError

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        logger.info("[%s] starting", self.name)
        result = self.run(state)
        logger.info("[%s] finished, updated keys=%s", self.name, list(result.keys()))
        return result
