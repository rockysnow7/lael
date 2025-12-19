from abc import ABC, abstractmethod
from common.constants import BUFFER_SIZE
from common.effect import Effect
from common.event import Event, Press, Release
from pydantic import BaseModel
from typing import Callable, Literal

import math
import torch


OscillatorFunction = Callable[[torch.Tensor, float, float, float], torch.Tensor]
"""(buffer, frequency, amplitude, phase) -> samples"""

OSCILLATOR_FUNCTIONS: dict[str, OscillatorFunction] = {
    "sine": lambda buffer, frequency, amplitude, phase: amplitude * torch.sin(2 * math.pi * frequency * buffer + phase),
    "square": lambda buffer, frequency, amplitude, phase: amplitude * torch.sign(torch.sin(2 * math.pi * frequency * buffer + phase)),
}
OscillatorFunctionName = Literal[*OSCILLATOR_FUNCTIONS.keys()]


class ADSREnvelope(BaseModel):
    attack_duration: float
    decay_duration: float
    sustain_volume: float
    release_duration: float
    __last_event: Event | None = None

    def process_event(self, event: Event) -> None:
        self.__last_event = event

    def generate_volume_multipliers(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate a tensor of volume multipliers over the given time buffer.

        Args:
            buffer: A tensor of shape `(BUFFER_SIZE,)`, where each element is the global time in seconds.

        Returns:
            A tensor of shape `(BUFFER_SIZE,)`.
        """

        if self.__last_event is None:
            return torch.zeros_like(buffer)

        volume_multipliers = torch.zeros_like(buffer)

        if isinstance(self.__last_event.event_content, Press):
            attack_end_time = self.__last_event.timestamp + self.attack_duration
            attack_mask = buffer < attack_end_time
            volume_multipliers[attack_mask] = (buffer[attack_mask] - self.__last_event.timestamp) / (attack_end_time - self.__last_event.timestamp)

            decay_end_time = attack_end_time + self.decay_duration
            decay_mask = (attack_end_time <= buffer) & (buffer < decay_end_time)
            volume_multipliers[decay_mask] = 1 - (buffer[decay_mask] - attack_end_time) / (decay_end_time - attack_end_time) * (1 - self.sustain_volume)

            sustain_mask = buffer >= decay_end_time
            volume_multipliers[sustain_mask] = self.sustain_volume

        elif isinstance(self.__last_event.event_content, Release):
            release_end_time = self.__last_event.timestamp + self.release_duration
            release_mask = buffer < release_end_time
            volume_multipliers[release_mask] = self.sustain_volume * (1 - (buffer[release_mask] - self.__last_event.timestamp) / (release_end_time - self.__last_event.timestamp))

            volume_multipliers[buffer >= release_end_time] = 0

        return volume_multipliers

class SoundSource(ABC, BaseModel):
    __effects: dict[str, Effect] = {}
    _last_event: Event | None = None
    adsr_envelope: ADSREnvelope | None = None

    def process_event(self, event: Event) -> None:
        self._last_event = event
        if self.adsr_envelope is not None:
            self.adsr_envelope.process_event(event)

    def add_effect(self, effect_name: str, effect: Effect) -> None:
        self.__effects[effect_name] = effect

    def delete_effect(self, effect_name: str) -> None:
        del self.__effects[effect_name]

    def __apply_effects(self, samples: torch.Tensor) -> torch.Tensor:
        for effect in self.__effects.values():
            samples = effect.apply(samples)
        return samples

    def generate_samples(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate samples over the given time buffer, applying the ADSR envelope and effects.

        Args:
            buffer: A tensor of shape `(BUFFER_SIZE,)`, where each element is the global time in seconds.

        Returns:
            A tensor of shape `(BUFFER_SIZE,)`.
        """

        samples = self._generate_samples_inner(buffer)

        if self.adsr_envelope is not None:
            volume_multipliers = self.adsr_envelope.generate_volume_multipliers(buffer)
        else:
            volume_multipliers = torch.ones(BUFFER_SIZE, device=buffer.device)
        samples = samples * volume_multipliers

        samples = self.__apply_effects(samples)
        return samples

    @abstractmethod
    def _generate_samples_inner(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Generate samples over the given time buffer.

        Args:
            buffer: A tensor of shape `(BUFFER_SIZE,)`, where each element is the global time in seconds.

        Returns:
            A tensor of shape `(BUFFER_SIZE,)`.
        """

class Oscillator(SoundSource):
    function_name: OscillatorFunctionName
    amplitude: float = 1.0
    phase: float = 0.0
    frequency: float = 0.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _generate_samples_inner(self, buffer: torch.Tensor) -> torch.Tensor:
        f = OSCILLATOR_FUNCTIONS[self.function_name]
        samples = f(buffer, self.frequency, self.amplitude, self.phase)

        return samples

    def process_event(self, event: Event) -> None:
        super().process_event(event)

        # if isinstance(event.event_content, Press) and event.event_content.frequency is not None:
        #     self.frequency = event.event_content.frequency

        match event.event_content:
            case Press(frequency=frequency):
                print(f"Press: {frequency=}")
                if frequency is not None:
                    self.frequency = frequency
            case Release():
                pass
            case _:
                raise ValueError(f"Invalid event content: {event.event_content=}")
