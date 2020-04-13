class Table:
    id: str
    title: [str]
    num_cols: int
    numeric_columns: [int]
    page_title: str
    num_data_rows: int
    second_title: str
    num_header_rows: int
    caption: str
    data: [[str]]

    def __init__(self, id, title, numCols, numericColumns, pgTitle,
                 numDataRows, secondTitle, numHeaderRows, caption, data):
        """Maps the data with camelCase from the json to python style
        snake case objects in this type."""
        self.id = id
        self.title = title
        self.num_cols = numCols
        self.numeric_columns = numericColumns
        self.page_title = pgTitle
        self.num_data_rows = numDataRows
        self.second_title = secondTitle
        self.num_header_rows = numHeaderRows
        self.caption = caption
        self.data = data

    def rows(self):
        """Is the same as .data but more explicit in what it is."""
        return self.data

    def empty_cell_count(self) -> int:
        """
        Counts table cells with no value in table
        :return: Empty cell count
        """
        empty_cells = 0
        for row in self.data:
            for cell in row:
                if not cell:
                    empty_cells += 1

        return empty_cells

    def column_tokens(self, col: int) -> [str]:
        """
        Returns the values in the given column of the table, combined and tokenized
        :param col: Column index
        :return: A list of tokens from values in the left-most column of the table
        """
        return [token for row in self.data if len(row) > col for token in row[col].split()]

    def header_tokens(self) -> [str]:
        return [token for header in self.title for token in header.split()]

    def body_tokens(self) -> [str]:
        """
        Returns the complete body of the table (all cells), combined and tokenized
        :return: The complete body of the table (all cells), combined and tokenized
        """

        return [token for row in self.data for cell in row for token in cell.split()]

class Query:
    id: int
    query: str

    def __init__(self, id, query):
        self.id = id
        self.query = query
