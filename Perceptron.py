class Perceptron:

    s = 0

    def __init__(self, b: int):
        self.b = b

    def compute(self, seq: list) -> bool:
        return sum(map(lambda x: int(x), seq)) > self.b
