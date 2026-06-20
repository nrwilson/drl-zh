from classes.cell import Cell


class Grid:
    """A Grid defines the geometry of the world and location of start, target, etc."""

    def __init__(self, spec: list[str]):
        """Creates the grid.

        It accepts a list of strings identifying the cell types, for example:
        [
            'EET',
            'EEE',
            'SEE'
        ]
        The start cell is at position (x=0, y=0), the target cell at position (x=2, y=2).
        """
        # Set the height and the width of the grid. Feel free to assume a well-formed input
        #       (i.e., the grid is rectangular and all strings have the same length).
        self.height: int = len(spec)
        self.width: int = len(spec[0])
        # Store a list of lists of `Cell`s, parsed from the input `spec`.
        cells = [[Cell(char) for char in row] for row in reversed(spec)]
        self.cells: list[list[Cell]] = cells

    def __str__(self) -> str:
        """Returns the string representation of the grid.

        For the example above in the __init__ method, the string representation looks like:
        E E T
        E E E
        S E E
        """
        # Return the string representation.
        rows = []
        for row in reversed(self.cells):
            rows.append(" ".join(row))
        result = "\n".join(rows)
        print(result)
        return result

    def __getitem__(self, key) -> Cell:
        """Returns what is in the cell at position defined by key=(x, y)."""
        # Return the Cell at position (x, y). Make sure the coordinates match the convention
        #       described in the __init__ method!!!
        x = key[0]
        y = key[1]
        return self.cells[y][x]