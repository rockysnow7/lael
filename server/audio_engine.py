from __future__ import annotations
from common.constants import SAMPLE_RATE, NUM_SECS_PER_BUFFER, BUFFER_SIZE
from common.effect import Effect
from common.sound_source import SoundSource, SoundSourceInput
from typing import Callable

import torch


device = torch.device("mps")


class SoundSourceGroup:
    def __init__(self) -> None:
        self.__children: dict[str, SoundSource | SoundSourceGroup] = {}
        self.__effects: dict[str, Effect] = {}

    def print_tree(self, indent: int = 0) -> None:
        spaces = " " * indent * 2
        for name, child in self.__children.items():
            print(f"{spaces}{name} ({child.__class__.__name__}):")
            if isinstance(child, SoundSourceGroup):
                child.print_tree(indent + 1)
            else:
                print(f"{spaces}{child}")

    def __getitem__(self, name: str) -> SoundSource | SoundSourceGroup:
        return self.__children[name]

    def get_child(self, name: str) -> SoundSource | SoundSourceGroup | None:
        return self.__children.get(name)

    def count_children(self, filter_function: Callable[[SoundSource | SoundSourceGroup], bool] = lambda _: True) -> int:
        count = len([child for child in self.__children.values() if filter_function(child)])
        count += sum(child.count_children(filter_function) for child in self.__children if isinstance(child, SoundSourceGroup))
        return count

    def add_child(self, name: str, child: SoundSource | SoundSourceGroup) -> None:
        self.__children[name] = child

    def delete_child(self, name: str) -> None:
        if name not in self.__children:
            raise KeyError(f"Child with name `{name}` not found")
        del self.__children[name]

    def add_effect(self, effect_name: str, effect: Effect) -> None:
        self.__effects[effect_name] = effect

    def delete_effect(self, effect_name: str) -> None:
        del self.__effects[effect_name]

    def __apply_effects(self, buffer: torch.Tensor) -> torch.Tensor:
        for effect in self.__effects.values():
            buffer = effect.apply(buffer)
        return buffer

    def process_input(self, input_: SoundSourceInput) -> None:
        for child in self.__children.values():
            child.process_input(input_)

    def generate_samples(self, buffer: torch.Tensor) -> torch.Tensor:
        if not self.__children:
            return torch.zeros_like(buffer)

        child_buffers = [
            child.generate_samples(buffer)
            for child in self.__children.values()
        ]
        buffer = torch.stack(child_buffers).sum(dim=0)
        return self.__apply_effects(buffer)


class AudioEngine:
    def __init__(self) -> None:
        self.__sound_sources_root = SoundSourceGroup()
        self.__time = 0.0

    def get_time(self) -> float:
        return self.__time

    def print_tree(self) -> None:
        print(f". ({self.__sound_sources_root.__class__.__name__}):")
        self.__sound_sources_root.print_tree(1)

    def __get_node_at_path(self, path: list[str], create_missing_groups: bool = False) -> SoundSource | SoundSourceGroup | None:
        current_node = self.__sound_sources_root
        for node_name in path:
            next_node = current_node.get_child(node_name)
            if next_node is None:
                if not create_missing_groups:
                    return None
                next_node = SoundSourceGroup()
                current_node.add_child(node_name, next_node)
            current_node = next_node
        return current_node

    def add_child_at_path(self, child: SoundSource | SoundSourceGroup, path: list[str]) -> None:
        name = path.pop()

        node = self.__get_node_at_path(path, create_missing_groups=True)
        if isinstance(node, SoundSourceGroup):
            node.add_child(name, child)
        else:
            raise ValueError(f"Node at path `{path}` is not a group")

    def delete_child_at_path(self, name: str, path: list[str]) -> None:
        node = self.__get_node_at_path(path)
        if isinstance(node, SoundSourceGroup):
            node.delete_child(name)
        else:
            raise ValueError(f"Node at path `{path}` is not a group")

    def add_effect_at_path(self, effect: Effect, path: list[str]) -> None:
        name = path.pop()
        node = self.__get_node_at_path(path)
        node.add_effect(name, effect)

    def delete_effect_at_path(self, path: list[str]) -> None:
        name = path.pop()
        node = self.__get_node_at_path(path)
        node.delete_effect(name)

    def send_input_to_node(self, input_: SoundSourceInput, path: list[str]) -> None:
        input_.timestamp = self.__time

        node = self.__get_node_at_path(path)
        if node is None:
            raise ValueError(f"Node at path `{path}` cannot process inputs")
        node.process_input(input_)

    def next_buffer(self) -> torch.Tensor:
        time_buffer = torch.arange(BUFFER_SIZE, device=device) / SAMPLE_RATE + self.__time
        root_buffer = self.__sound_sources_root.generate_samples(time_buffer)

        empty_buffer = torch.zeros(BUFFER_SIZE, device=device)
        added_buffer = torch.stack([empty_buffer, root_buffer]).sum(dim=0)

        self.__time += NUM_SECS_PER_BUFFER

        return added_buffer
