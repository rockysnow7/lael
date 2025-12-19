import numpy as np
import pyaudio
import threading

from common.constants import NUM_SECS_PER_BUFFER, SAMPLE_RATE, BUFFER_SIZE
from client.utils import add_effect_to_sound_source, generate_next_buffer, add_oscillator, Note, send_event_to_node
from common.effect import Volume
from common.sound_source import ADSREnvelope, Oscillator, Event, Press, Release


def add_chord(group_name: str, notes: list[Note], envelope: ADSREnvelope | None = None) -> None:
    for i, note in enumerate(notes):
        oscillator = Oscillator(function_name="sine", frequency=float(note))
        add_oscillator(f"{group_name}/{i}", oscillator, envelope)


class Client:
    def __init__(self) -> None:
        self.__buffers: list[bytes] = []
        self.__time = 0.0

        p = pyaudio.PyAudio()
        self.__stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
        )

    def write_one_buffer(self, empty: bool = False) -> None:
        """
        Writes a buffer to the stream.

        Args:
            empty: If `True`, writes one second of silence. If `False` (default), writes the next buffer from the server.
        """

        if empty:
            samples = np.zeros(BUFFER_SIZE, dtype=np.float32).tobytes()
            self.__buffers.append(samples)
            return

        samples = generate_next_buffer()
        self.__buffers.append(samples)

    def play_thread(self) -> None:
        """Plays the buffers in the stream."""

        while True:
            if len(self.__buffers) <= 3:
                self.write_one_buffer()

            samples = self.__buffers.pop(0)
            self.__stream.write(samples)
            self.__time += NUM_SECS_PER_BUFFER

    def send_thread(self) -> None:
        """Sends the inputs to the server."""

        while True:
            event_name = input("> ")
            parts = event_name.split()
            if parts[0] == "press":
                if len(parts) == 2:
                    note_name = parts[1]
                    frequency = float(Note.from_name(note_name))
                else:
                    frequency = None
                event = Event(event_content=Press(frequency=frequency))
            elif parts[0] == "rel":
                event = Event(event_content=Release())
            else:
                print("Invalid event")
                continue

            send_event_to_node("Cmaj", event)

    def run(self) -> None:
        """Runs the client."""

        threading.Thread(target=self.play_thread).start()
        threading.Thread(target=self.send_thread).start()


# add_oscillator(
#     "a",
#     Oscillator(function_name="sine"),
#     ADSREnvelope(attack_duration=0.1, decay_duration=0.1, sustain_volume=0.8, release_duration=0.1),
# )
add_chord(
    "Cmaj",
    [Note.from_name("C3"), Note.from_name("E3"), Note.from_name("G3")],
    ADSREnvelope(attack_duration=0.1, decay_duration=0.1, sustain_volume=0.8, release_duration=0.1),
)
add_effect_to_sound_source(
    "Cmaj",
    Volume(volume=0.1),
)

client = Client()
client.run()
