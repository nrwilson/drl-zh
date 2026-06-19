from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """A state in the grid world is an x, y position."""

    x: int = 0
    y: int = 0

    def pos(self) -> tuple[int]:
        return (self.x, self.y)
