class Table:
    id: str
    title: [str]
    numCols: int
    numericColumns: [int]
    pgTitle: str
    numDataRows: int
    secondTitle: str
    numHeaderRows: int
    caption: str
    data: [[str]]

    def rows(self):
        return self.data


class Query:
    query: str

    def terms(self):
        return self.query.split(" ")


class TermVector:
    terms: [str]
