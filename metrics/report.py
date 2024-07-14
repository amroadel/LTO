import tabulate

class Report:

    def __init__(self, 
        headers: list[str], 
        logfmt: str = 'simple_outline', 
        savefmt: str = 'tsv',
        logidx: int = 6 # Only log the first 6 column on terminal
    ):
        self.headers = headers
        self.logfmt = logfmt
        self.savefmt = savefmt
        if logidx == -1:
            logidx = len(self.headers)
        self.logidx = logidx

    def stringfy(self, data) -> str:
        return tabulate.tabulate(
            [ d[:self.logidx] for d in data],
            headers=self.headers[:self.logidx],
            tablefmt=self.logfmt,
            floatfmt='.2f'
        )

    def save(self, data, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(tabulate.tabulate(
                data,
                headers=[],
                tablefmt=self.savefmt,
                floatfmt='.2f'
            ))
