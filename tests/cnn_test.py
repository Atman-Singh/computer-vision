import pytest
from pathlib import Path
import sys
import os
import torch
from datasets import load_from_disk, load_dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from infrastructure.cnn import ConvolutionalNeuralNetwork


ROOT_DIR = Path(__file__).resolve().parent.parent

ds_dir = ROOT_DIR / "data" / "input" / "mnist"
os.makedirs(ds_dir.parent, exist_ok=True)
if ds_dir.is_dir():
    ds = load_from_disk(ds_dir)
else:
    ds = load_dataset("ylecun/mnist")
    ds.save_to_disk(ds_dir)

def T(x): 
    return torch.tensor(x, device='cuda')

@pytest.mark.parametrize(
    "matrix,kernel,step,expected",
    [
        # 1) 1x1 kernel (identity)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[1]],
            1,
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
        ),

        # 2) 1x1 kernel (scale by 2)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[2]],
            1,
            [[2,4,6],
             [8,10,12],
             [14,16,18]],
        ),

        # 3) 2x2 all-ones (sliding sum)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[1,1],
             [1,1]],
            1,
            [[12,16],
             [24,28]],
        ),

        # 4) 2x2 top-left picker (extracts top-left of each window)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[1,0],
             [0,0]],
            1,
            [[1,2],
             [4,5]],
        ),

        # 5) 2x2 all-ones with stride 2
        (
                [[ 1, 2, 3, 4],
                [ 5, 6, 7, 8],
                [ 9,10,11,12],
                [13,14,15,16]],
                [[1,1],
                [1,1]],
                1,
            [[14, 18, 22],
            [30, 34, 38],
            [46, 50, 54]],
        ),

        # 6) 3x3 all-ones over 3x3 (single output)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[1,1,1],
             [1,1,1],
             [1,1,1]],
            1,
            [[45]],
        ),

        # 7) 3x3 center-identity (picks center of 3x3 window)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[0,0,0],
             [0,1,0],
             [0,0,0]],
            1,
            [[5]],
        ),

        # 8) negatives cancel
        (
            [[ 1,-1],
             [ 2,-2]],
            [[1,1],
             [1,1]],
            1,
            [[0]],
        ),

        # 9) 3x3 diagonal kernel on 4x4 (2x2 output)
        (
            [[ 1, 2, 3, 4],
             [ 5, 6, 7, 8],
             [ 9,10,11,12],
             [13,14,15,16]],
            [[1,0,0],
             [0,1,0],
             [0,0,1]],
            1,
            [[18,21],
             [30,33]],
        ),

        # 10) vertical edge-like kernel on 3x3 (single output)
        (
            [[1,2,3],
             [4,5,6],
             [7,8,9]],
            [[ 1,0,-1],
             [ 1,0,-1],
             [ 1,0,-1]],
            1,
            [[-6]],
        ),
    ],
)

def test_traverse_matrix(matrix, kernel, step, expected):
    cnn = ConvolutionalNeuralNetwork(pool_size=(2,2), ds=ds, learning_rate=1)
    out = cnn._traverse_matrix(matrix=T(matrix), kernel=T(kernel), step=step)
    assert torch.allclose(out, T(expected))
