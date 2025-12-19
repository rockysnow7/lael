from server.audio_engine import AudioEngine
from common.sound_source import Oscillator, SoundSource, Event
from common.effect import Effect, Volume, Vibrato
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

@app.post("/send-event-to-node")
def send_event_to_node(
    event: Event = Body(title="The event to send"),
    path: str = Body(title="The path of the node to send the event to"),
) -> None:
    path = path.split("/")
    audio_engine.send_event_to_node(event, path)

sound_sources: list[type[SoundSource]] = [Oscillator]
for sound_source in sound_sources:
    api_path = f"/add/sound-source/{sound_source.__name__.lower()}"

    @app.put(api_path)
    def add_sound_source(
        source: sound_source = Body(title=f"The {sound_source.__name__} to add"),
        path: str = Body(title=f"The path at which to add the {sound_source.__name__}"),
    ) -> None:
        path = path.split("/")
        audio_engine.add_child_at_path(source, path)

effects: list[type[Effect]] = [Volume, Vibrato]
for effect in effects:
    path = f"/effect/add/{effect.__name__.lower()}"

    @app.put(path)
    def add_effect(
        effect: effect = Body(title=f"The {effect.__name__} to add"),
        path: str = Body(title=f"The path of the node to which the {effect.__name__} should be added"),
    ) -> None:
        path = path.split("/")
        audio_engine.add_effect_at_path(effect, path)
