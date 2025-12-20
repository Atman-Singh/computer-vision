import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset, DatasetDict
from infrastructure.convolutional_neural_network import ConvolutionalNeuralNetwork

def main():
    ds = load_dataset("ylecun/mnist")
    cnn = ConvolutionalNeuralNetwork(28, 28, ds)
    cnn.forward


if __name__ == '__main__':
    main()