from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Callable, Literal

import math
import torch


class Effect(ABC, BaseModel):
    @abstractmethod
    def apply(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Apply the effect to the given buffer of samples.

        Args:
            buffer: A tensor of shape `(BUFFER_SIZE,)`, where each element is a sample.

        Returns:
            A tensor of shape `(BUFFER_SIZE,)`.
        """

class Volume(Effect):
    volume: float

    def apply(self, buffer: torch.Tensor) -> torch.Tensor:
        return buffer * self.volume

class Vibrato(Effect):
    frequency: float
    depth: float

    ...
