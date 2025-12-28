import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset, load_from_disk
from infrastructure.convolutional_neural_network import ConvolutionalNeuralNetwork

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    ds_dir = ROOT_DIR / "data" / "input" / "mnist"
    os.makedirs(ds_dir.parent, exist_ok=True)
    if ds_dir.is_dir():
        ds = load_from_disk(ds_dir)
    else:
        ds = load_dataset("ylecun/mnist")
        ds.save_to_disk(ds_dir)
    cnn = ConvolutionalNeuralNetwork(28, 28, ds, 0.3)
    cnn.train(1)
    
if __name__ == '__main__':
    main()