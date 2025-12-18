from server.audio_engine import AudioEngine
from common.sound_source import Oscillator, SoundSourceInput
from common.effect import Volume, Vibrato
from fastapi import FastAPI, Body


audio_engine = AudioEngine()
app = FastAPI()


@app.get("/get-time")
def get_time() -> float:
    return audio_engine.get_time()

@app.post("/generate-next-buffer")
def generate_next_buffer() -> list[float]:
    samples = audio_engine.next_buffer()
    return samples.tolist()

@app.post("/send-input-to-node")
def send_input_to_node(
    input: SoundSourceInput = Body(title="The input to send"),
    path: str = Body(title="The path of the node to send the input to"),
) -> None:
    path = path.split("/")
    audio_engine.send_input_to_node(input, path)

sound_sources = [Oscillator]
for sound_source in sound_sources:
    api_path = f"/add/sound-source/{sound_source.__name__.lower()}"

    @app.put(api_path)
    def add_sound_source(
        source: sound_source = Body(title=f"The {sound_source.__name__} to add"),
        path: str = Body(title=f"The path at which to add the {sound_source.__name__}"),
    ) -> None:
        path = path.split("/")
        audio_engine.add_child_at_path(source, path)

# effects = [Volume, Vibrato]
# for effect in effects:
#     path = f"/effect/add/{effect.__name__.lower()}"

#     @app.put(path)
#     def add_effect(
#         sound_source_name: str = Body(...),
#         effect_name: str = Body(...),
#         effect: effect = Body(...),
#     ) -> None:
#         audio_engine.add_effect_to_sound_source(sound_source_name, effect_name, effect)
