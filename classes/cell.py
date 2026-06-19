from enum import StrEnum


class Cell(StrEnum):
    """A cell is what is in the grid at a given position."""

    START = "S"
    TARGET = "T"
    EMPTY = "E"
    WALL = "W"
    BOMB = "B"