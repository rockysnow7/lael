import numpy as np
import pyaudio
import threading

from common.constants import NUM_SECS_PER_BUFFER, SAMPLE_RATE, BUFFER_SIZE
from client.utils import add_effect_to_sound_source, generate_next_buffer, add_oscillator, Note, send_input_to_node
from common.effect import Volume
from common.sound_source import ADSREnvelope, Oscillator, SoundSourceInput, SoundSourcePressDown, SoundSourcePressUp


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
            # if not self.__input_queue:
            #     continue

            # next_input = self.__input_queue[0]
            # if self.__time < next_input.timestamp:
            #     continue

            # self.__input_queue.pop(0)
            # send_input_to_node(next_input.path, next_input.input)

            input_ = input("> ")
            parts = input_.split()
            if parts[0] == "press":
                frequency = float(Note.from_name(parts[1]))
                input_ = SoundSourcePressDown(frequency=frequency)
            elif parts[0] == "stop":
                input_ = SoundSourcePressUp()
            else:
                print("Invalid input")
                continue

            send_input_to_node("a", input_)

    def run(self) -> None:
        """Runs the client."""

        threading.Thread(target=self.play_thread).start()
        threading.Thread(target=self.send_thread).start()


add_oscillator(
    "a",
    Oscillator(function_name="sine"),
    ADSREnvelope(attack_duration=0.1, decay_duration=0.1, sustain_volume=0.8, release_duration=0.1),
)
add_effect_to_sound_source(
    "a",
    Volume(volume=0.1),
)

client = Client()
client.run()
