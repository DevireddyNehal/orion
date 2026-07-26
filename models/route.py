from dataclasses import dataclass


@dataclass
class Route:
    model: str
    capabilities: list[str]