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


class Query:
    id: int
    query: str

    def __init__(self, id, query):
        self.id = id
        self.query = query
