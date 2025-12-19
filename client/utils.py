from __future__ import annotations
from common.effect import Effect
from common.sound_source import ADSREnvelope, Oscillator, Event

import requests
import numpy as np


class Note:
    def __init__(self, name: str, frequency: float) -> None:
        self.name = name
        self.frequency = frequency

    @staticmethod
    def from_name(name: str) -> Note:
        """
        Returns the frequency of the note.

        Args:
            name: The name of the note in the format <letter><|#><octave>.

        Returns:
            The frequency of the note.
        """

        name = name.upper()

        if len(name) not in {2, 3}:
            raise ValueError(f"Invalid note name: {name}")
        parts = list(name)

        accidental = ""
        if len(parts) == 3:
            accidental = parts.pop(1)

        letter, octave = parts
        octave = int(octave)
        note = letter + accidental

        all_notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        semitones_from_c = all_notes.index(note)
        octaves_from_c4 = octave - 4
        semitones_from_c4 = semitones_from_c + octaves_from_c4 * 12
        c4_frequency = 261.63
        frequency = c4_frequency * 2 ** (semitones_from_c4 / 12)

        return Note(name, frequency)

    def __float__(self) -> float:
        return self.frequency

    def plus_semitones(self, semitones: int) -> Note:
        ... # TODO

    def major_chord(self) -> list[Note]:
        return [self, self.plus_semitones(4), self.plus_semitones(7)]

    def minor_chord(self) -> list[Note]:
        return [self, self.plus_semitones(3), self.plus_semitones(7)]


def generate_next_buffer() -> bytes:
    samples = requests.post("http://localhost:8000/generate-next-buffer").json()
    samples = np.array(samples, dtype=np.float32).tobytes()
    return samples

def add_oscillator(
    path: str,
    oscillator: Oscillator,
    adsr_envelope: ADSREnvelope | None = None,
) -> None:
    if adsr_envelope is not None:
        oscillator.adsr_envelope = adsr_envelope

    requests.put(
        "http://localhost:8000/add/sound-source/oscillator",
        json={
            "source": oscillator.model_dump(),
            "path": path,
        },
    )

def add_effect_to_sound_source(
    path: str,
    effect: Effect,
) -> None:
    requests.put(
        f"http://localhost:8000/effect/add/{effect.__class__.__name__.lower()}",
        json={
            "effect": effect.model_dump(),
            "path": path,
        },
    )

def send_event_to_node(
    path: str,
    event: Event,
) -> None:
    print(f"Sending event {event=} as {event.model_dump()}")
    requests.post(
        "http://localhost:8000/send-event-to-node",
        json={
            "event": event.model_dump(),
            "path": path,
        },
    )

def get_time() -> float:
    return requests.get("http://localhost:8000/get-time").json()
