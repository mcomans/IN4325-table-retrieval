class Table:
    data: any

    def terms(self):
        # TODO: Implementation
        pass


class Query:
    query: str

    def terms(self):
        return self.query.split(" ")


class TermVector:
    terms: [str]
