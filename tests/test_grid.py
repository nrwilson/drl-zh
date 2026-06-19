from classes.grid import Grid
from classes.cell import Cell


def test_grid():
    grid = Grid(
        [
            "EEET",
            "EWEB",
            "SEEE",
        ]
    )
    assert grid[3, 1] == Cell.BOMB
    assert grid[0, 0] == Cell.START
    assert grid.height == 3
    assert grid.width == 4
    return grid