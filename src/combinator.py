from torch import nn


class Combinator(nn.Module):
    def __init__(self, layers_neurons: list[int]) -> None:
        super(Combinator, self).__init__()
        _classifier = nn.Sequential()
        for inp, out in zip(layers_neurons, layers_neurons[1:]):
            _classifier.append(nn.Linear(inp, out))
            _classifier.append(nn.ReLU(inplace=True))

        _classifier.append(nn.Softmax(dim=-1))
        self.classifier = _classifier

    def forward(self, x):
        return self.classifier(x)
