import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset, load_from_disk
from safetensors.torch import load_file

from infrastructure.cnn import ConvolutionalNeuralNetwork

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    ds_dir = ROOT_DIR / "data" / "input" / "mnist"
    os.makedirs(ds_dir.parent, exist_ok=True)
    if ds_dir.is_dir():
        ds = load_from_disk(ds_dir)
    else:
        ds = load_dataset("ylecun/mnist")
        ds.save_to_disk(ds_dir)

    tensors_dir = ROOT_DIR / "data" / "output" / "cnn" / "fresh"
    state = load_file(tensors_dir / "model.safetensors", device="cuda")

    kernel_stacks = [state[f'layer1.kernels'], state[f'layer2.kernels']]
    biases = [state[f'layer1.bias'], state[f'layer2.bias'], state[f'layer3.bias']]
    weights = state[f'fc.weights']

    cnn = ConvolutionalNeuralNetwork(pool_size=(2,2), ds=ds, learning_rate=0.3, kernel_stacks=kernel_stacks, biases=biases, weights=weights)
    print(cnn.test_inference())
    
if __name__ == '__main__':
    main()