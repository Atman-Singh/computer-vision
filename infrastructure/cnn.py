from datasets import DatasetDict
import torch
import torch.nn.functional as F
from torch.func import vmap
import time
import logging
from logging import Logger
from pathlib import Path
import os
import sys
from safetensors.torch import save_file

ROOT_DIR = Path(__file__).resolve().parent.parent

class ConvolutionalNeuralNetwork:

    def __init__(self, pool_size: tuple, ds: DatasetDict, learning_rate: float, learning_rate_decay: float | None = None,
                 decay_step: int | None = None,
                 kernel_stacks: list | None = None, biases: list | None = None, weights: torch.Tensor | None = None, 
                 activation_functions: list | None = None, logger: Logger | None = None,
                 ):
        # logging
        if logger:
            self.logger = logger
        else:
            level = logging.INFO

            logs_dir = ROOT_DIR / "logs"
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(filename=str(logs_dir / f'{__name__}.log'), level=level)

            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level) # Set the console handler's level

            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            ch.setFormatter(formatter)

            self.logger.addHandler(ch)

        self.logger.info('Logger initialized succesfully')

        if torch.cuda.is_available():
            self.logger.info("CUDA is available. PyTorch can use the GPU.")
            self.logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            self.logger.info(f"Current GPU name: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"CUDA Version built with PyTorch: {torch.version.cuda}")
        else:
            self.logger.warning("CUDA is not available. PyTorch will use the CPU.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ds = ds
        labels = set()
        for l in ds['train']['label']:
            labels.add(l)
        self.output_range = len(labels)
        
        # convert pngs to tensors containing pixel values
        self.rows = len(ds['train'])
        height, width = ds['train']['image'][0].size
        if width != height:
            self.logger.info('TODO: support for non-square images')
        self.images = torch.empty((self.rows, width, height), dtype=torch.float32, device=self.device)
        for i, row in enumerate(ds['train'].select(range(self.rows))):
            self.images[i] = torch.tensor(
                list(row['image'].getdata()), 
                device=self.device, 
                dtype=torch.float32
            ).reshape(width, height) / 255.0

        self.pool_size = pool_size

        self.learning_rate, self.learning_rate_decay, self.decay_step = learning_rate, learning_rate_decay, decay_step

        # define kernels
        torch.manual_seed(3500)
        if kernel_stacks:
            self.kernel_stacks = kernel_stacks
        else:
            min_fence = -0.2
            max_fence = 0.2

            # initialize kernels with random values and transform them to range [-0.2, 0.2)
            # 0: # of output channels
            # 1: # of inputs per output channel
            # 2, 3: rows, cols
            # num of kernels = inputs * outputs
            kernel_l1 = torch.rand(2, 1, 5, 5, device=self.device) * (max_fence - min_fence) + min_fence
            kernel_l2 = torch.rand(2, 2, 3, 3, device=self.device) * (max_fence - min_fence) + min_fence
            self.kernel_stacks = [kernel_l1, kernel_l2]
        
        # initialize biases
        if biases:
            self.biases = biases
        else:
            bias_l1 = torch.rand(2, 1, device=self.device) * (max_fence - min_fence) + min_fence
            bias_l2 = torch.rand(2, 2, device=self.device) * (max_fence - min_fence) + min_fence
            self.biases = [bias_l1, bias_l2]

        # initialize fully connected layer weights
        self.weights = weights

        # activation functions
        if activation_functions:
            self.activation_functions = activation_functions
        else:
            self.activation_functions = ['relu', 'relu']
    
    def _inference(self, data: torch.Tensor, i: int) -> torch.Tensor:
        output_v = self.forward(
            data, 
            self.kernel_stacks, 
            self.weights, 
            self.biases, 
            self.activation_functions, 
            self.pool_size,
            i
        )[-1][-1]
        return output_v.max(dim=0).indices

    def test_inference(self, i: int | None = None) -> float:
        if i is None:
            i = len(self.images)
        labels = torch.tensor(self.ds['train'][:i]['label'], device=self.device)
        output = self._inference(self.images[:i], i)
        truth = labels == output
        score = truth.sum() / i
        return round(score.item(), 4)

    def train(self, epochs: int, images_processed: int | None = None):
        torch.cuda.synchronize()
        start = time.perf_counter()

        if images_processed is None:
            images_processed = self.rows
        input_x = self.images[:images_processed]
        prev_cost, cost = None, None
        for i in range(epochs):
            if prev_cost is not None and cost > prev_cost:
                self.learning_rate *= self.learning_rate_decay 

            output_conv = self.forward(
                input_x, 
                self.kernel_stacks, 
                self.weights, self.biases, 
                self.activation_functions, 
                self.pool_size, 
                images_processed
            )
            # append initialized fully connected biases
            if len(self.biases) == len(self.kernel_stacks):
                self.biases.append(output_conv[-1][-2])

            self.logger.debug(output_conv[-1][1].shape)
            self.weights = output_conv[-1][1]

            prev_cost = cost
            cost = self._compute_cost(output_conv[-1][-1], self.ds)
            deltas = self.backward(output_conv, self.kernel_stacks, input_x, self.output_range, images_processed)
            self.kernel_stacks, self.biases, self.weights = self._get_updated_parameters(self.kernel_stacks, self.biases, self.weights, deltas)

            elapsed = time.perf_counter()-start
            min, sec = elapsed // 60, int(elapsed % 60)
            self.logger.info(f'Epoch {i+1}: cost reduced to {cost:4f} over {int(min)}min, {sec}s')
            # self.logger.info(
            #     f"grad norms: fcW={deltas[-1][0].norm().item():.3e} "
            #     f"k1={deltas[0][1].norm().item():.3e} k2={deltas[1][1].norm().item():.3e}"
            # )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        min, sec = elapsed // 60, int(elapsed % 60)
        self.logger.info(f'Finished CNN training in {int(min)}min, {sec}s')

        tensors_dir = ROOT_DIR / "data" / "output" / "cnn" / "fresh"
        os.makedirs(tensors_dir, exist_ok=True)

        # export weights & biases
        state = {}
        for i, kernel_stack in enumerate(self.kernel_stacks):
            state[f'layer{i+1}.kernels'] = kernel_stack
        for i, bias in enumerate(self.biases):
            state[f'layer{i+1}.bias'] = bias
        state['fc.weights'] = self.weights
        save_file(
            state,
            tensors_dir / "model.safetensors"
        )

        self.logger.info(f'Saved weights & biases to {str(tensors_dir)}')

    # update weights and biases using deltas returned by backpropagation
    def _get_updated_parameters(self, kernel_stacks: list, biases: list, weights: torch.Tensor, deltas: list) -> tuple:
        # convolutional pipeline updates
        updated_kernels, updated_biases = [], []
        for i in range(len(kernel_stacks)):
            _, kernel_d, bias_d = deltas[i]
            updated_kernels.append(kernel_stacks[i] - self.learning_rate * kernel_d)
            updated_biases.append(biases[i] - self.learning_rate * bias_d)
            
        # fully connected layer updates
        weights_d, bias_d = deltas[-1]
        updated_weights = weights - self.learning_rate * weights_d
        updated_biases.append(biases[-1] - self.learning_rate * bias_d.unsqueeze(-1))

        return (updated_kernels, updated_biases, updated_weights)

    # helper function to divide out rows from first dimension of tensor
    def _seperate(self, tensor: torch.Tensor, i: int):
        shape = tensor.shape
        if len(shape) > 1:
            return tensor.reshape(i, shape[0]//i, *shape[1:])
        return tensor.reshape(i, shape[0]//i)

    # computes convolution between kernel and matrix
    def _traverse_matrix(self, matrix: torch.Tensor, kernel: torch.Tensor, step: int) -> torch.Tensor:
        if len(kernel.shape) != 2:
            print(f"kernel has a rank of {len(kernel.shape)}")
        
        kh, kw = kernel.shape[-2], kernel.shape[-1]
        # create tensor of all windows kernel will slide over and repeat by number of kernels
        matrix_unfolded = (
            matrix
                .unfold(0, kh, step)
                .unfold(1, kw, step)
        )
        unfolded_sum = kernel * matrix_unfolded
        return unfolded_sum.sum(dim=(-1,-2))

    # computes maxpooling of matrix
    def _max_pool(self, matrix: torch.Tensor, pool_size: tuple) -> torch.Tensor:
        if len(matrix.shape) != 2:
            raise Exception(f"Matrix has a rank of {len(matrix.shape)}")
        if matrix.shape[0] % pool_size[0] != 0 or matrix.shape[1] % pool_size[1] != 0:
            raise Exception(f"Pool size {pool_size} is not a multiple of matrix shape {matrix.shape}")

        pool_h, pool_w = pool_size
        pooled_h, pooled_w = matrix.shape[0] // pool_h, matrix.shape[1] // pool_w
        
        return (
            matrix
            .reshape(pooled_h, pool_h, pooled_w, pool_w)
            .max(dim=3)
            .values
            .max(dim=1)
            .values
        )

    # makes prediction through forward propagation
    def forward(
            self, 
            data: torch.Tensor, 
            kernel_stacks: list, 
            weights: torch.Tensor, 
            biases: list, 
            activation_functions: list, 
            pool_size: tuple,
            images_processed: int
            ) -> list:
        if not (isinstance(data, torch.Tensor) or isinstance(data, list)):
            raise Exception(f"Data is not of type 'list', inputted type is: {type(data)}")
        if not isinstance(kernel_stacks, list):
            raise Exception(f"Kernels is not of type 'list', inputted type is: {type(kernel_stacks)}")
        if len(kernel_stacks) > len(biases):
            raise Exception(f'number of kernel stacks {len(kernel_stacks)} greater than number of biases ({len(biases)})')
        if len(kernel_stacks) != len(activation_functions):
            self.logger.warning(f"Number of kernel stacks inputted ({len(kernel_stacks)}) does " +
                f"not equal number of activation functions inputted " +
                f"({len(activation_functions)})")
        # define working input
        input_stack = data.unsqueeze(1)
        output = [None for _ in range(len(kernel_stacks)+1)]
        for i, kernel_stack in enumerate(kernel_stacks):
            # initialize activation func
            func = torch.nn.ReLU()
            if i < len(activation_functions):
                if activation_functions[i].lower() == 'sigmoid':
                    func = torch.nn.Sigmoid()
                elif activation_functions[i].lower() != 'relu':
                    raise Exception(f'Activation function "{activation_functions[i]}" ' +
                                f"is not a valid activation function.")
            # take convolution
            # img, kernel, step
            conv_single = vmap(self._traverse_matrix, in_dims=(0, 0, None))
            conv_over_kernels = vmap(conv_single, in_dims=(None,0,None))
            convolution = vmap(conv_over_kernels, in_dims=(0, None, None))(input_stack, kernel_stack, 1)

            self.logger.debug(convolution.shape, biases[i].shape)
            convolution = convolution + biases[i].unsqueeze(-1).unsqueeze(-1)
            # take activation
            activation = func(convolution)
            # pool activation
            pooled = self._seperate(
                vmap(self._max_pool, in_dims=(0,None))(activation.flatten(end_dim=2), pool_size),
                images_processed
            )

            # update output
            output[i] = (convolution, activation, pooled)
            # update working input
            input_stack = pooled

        # flattening
        flattened = pooled.flatten(1)

        # initialize weights and biases for fully connected layer if not already
        if weights is None and len(biases) == len(kernel_stacks):
            torch.manual_seed(3500)
            min_fence = -0.2
            max_fence = 0.2
            
            weights = torch.rand(self.output_range, flattened.shape[1], device=self.device) * (max_fence - min_fence) + min_fence
            biases.append(torch.rand(self.output_range, 1, device=self.device) * (max_fence - min_fence) + min_fence)

        # compute fully connected layer output
        logits = (weights @ flattened.T + biases[-1]).squeeze(0)
        output[-1] = (flattened, weights, biases[-1], logits)
        
        return output

    # binary cross-entropy loss function
    def _compute_loss(self, a: torch.Tensor, y: torch.Tensor):
        return -(y * torch.log(a) + (1 - y) * torch.log(1 - a))

    # function to compute cost of output for all images
    def _compute_cost(self, logits: torch.Tensor, dataset: DatasetDict):
        # TODO: code myself
        labels = torch.tensor(dataset['train'][:logits.shape[1]]['label'], device=self.device)
        loss = F.cross_entropy(logits.T, labels, reduction="mean")  
        return loss

    # compute delta for pooled matrix (dA/dL)
    def _max_pool_d(self, matrix: torch.Tensor, pooled_d: torch.Tensor, pool_size: tuple) -> torch.Tensor:
        if len(matrix.shape) != 2:
            raise Exception(f"Matrix has a rank of {len(matrix.shape)}")
        if len(pool_size) != 2:
            raise Exception(f"Pool size {pool_size} has a rank of {len(pool_size)}")
            
        pool_h, pool_w = pool_size
        if matrix.shape[0] % pool_h != 0 or matrix.shape[1] % pool_h != 0:
            raise Exception(f"Pool size {pool_size} is not a multiple of matrix shape {matrix.shape}")

        matrix_d = torch.zeros_like(matrix)
        for i in range(0, matrix.shape[0] - pool_h + 1, pool_h):
            for j in range(0, matrix.shape[1] - pool_w + 1, pool_w):
                window = matrix[i:i+pool_h, j:j+pool_w]
                _, idx = window.reshape(-1).max(dim=0)
                r, c = idx // window.size(1) + i, idx % window.size(1) + j

                matrix_d.index_put_(
                    (r, c), 
                    pooled_d[i // pool_h, j // pool_w],
                    accumulate=True
                )  

        return matrix_d

    # compute delta for post ReLU matrix
    def _relu_d(self, matrix: torch.Tensor):
        matrix_d = torch.zeros_like(matrix)
        matrix_d[matrix > 0] = 1
        return matrix_d

    # computes deltas through backward propogation
    def backward(
            self,
            forward_output: list, 
            kernel_stacks: list, 
            input_x: torch.Tensor, 
            output_range: int,
            images_processed: int
            ):
        # compute loss
        activation_fc = torch.softmax(forward_output[-1][3], dim=0)
        
        # compute fully connected layer deltas
        biases_fc_d = None
        flattened, weights = forward_output[-1][0], forward_output[-1][1]

        labels = torch.tensor(self.ds['train']['label'], dtype=torch.long, device=self.device)
        expected = torch.zeros(output_range, images_processed, device=self.device)
        expected[labels[:images_processed], torch.arange(images_processed)] = 1
        
        output_fc_d = activation_fc - expected #dZ
        weights_d = output_fc_d @ flattened / images_processed #dW, averaged
        biases_fc_d = output_fc_d.sum(dim=1) / images_processed #dB, averaged
        flattened_d = weights.T @ output_fc_d #dF, kept seperate
        
        # compute convolutional pass deltas, iterate backwards starting from last convolutional layer
        deltas = [None for _ in range(len(forward_output))]
        for i, _ in reversed(list(enumerate(forward_output[:-1]))):
            convolution, activation, pooled = (
                forward_output[i][0], 
                forward_output[i][1], 
                forward_output[i][2]
            )
            
            # compute delta for pooled matrix
            if i == len(kernel_stacks)-1:
                # if at last convolutional layer, compute delta based on flattened
                pooled_d = flattened_d.T.reshape(pooled.shape)
            else:
                # if at a middle convolutional layer, use past delta
                kernel, prev_kernel = kernel_stacks[i], kernel_stacks[i+1]
                prev_convolution_d = deltas[i+1][0]
                
                padding = (pooled.shape[-1] + prev_kernel.shape[-1] - prev_convolution_d.shape[-1] - 1) // 2
                prev_convolution_d_padded = F.pad(prev_convolution_d,(padding,)*4,value=0)
                rotated_kernel = (
                    prev_kernel
                    .rot90(2,(-2,-1))
                )

                pooled_d_vmap_single = vmap(self._traverse_matrix, in_dims=(0,0,None))
                pooled_d_vmap_channel = vmap(pooled_d_vmap_single, in_dims=(0,0,None))
                pooled_d = vmap(pooled_d_vmap_channel, in_dims=(0,None,None))(prev_convolution_d_padded, rotated_kernel, 1).sum(2)

            # compute deltas for layer 2 activations
            s = activation.shape[:3]
            activation_d = (
                vmap(self._max_pool_d, in_dims=(0, 0, None))(activation.flatten(end_dim=2), pooled_d.flatten(end_dim=1), (2,2))
                .reshape(*s, *activation.shape[3:])
            )

            # compute activation deltas for convolutionns
            convolution_d_activation = self._relu_d(convolution)
            # compute cost deltas for convolutions
            convolution_d = (convolution_d_activation * activation_d)

            if i == 0:
                # if at the beginning convolutional layer, input tensor is the input to the convolutional neuron
                input_t = input_x.unsqueeze(1)
            else:
                # next pooled matrix is input to convolutional neuron
                input_t = forward_output[i-1][2]
            # compute deltas for kernels, averaged and used to update kernels
            kernel_d_single = vmap(self._traverse_matrix, in_dims=(0,0,None))
            kernel_d_channel = vmap(kernel_d_single, in_dims=(None,0,None))
            kernel_d = torch.mean(
                vmap(kernel_d_channel, in_dims=(0,0,None))(input_t, convolution_d, 1),
                dim=0
            )

            # compute deltas for biases, averaged and used to update biases
            bias_d = convolution_d.sum((0,3,4)) / images_processed
            self.logger.debug(bias_d.shape)
            
            deltas[i] = (convolution_d, kernel_d, bias_d)
        deltas[-1] = (weights_d, biases_fc_d)
        
        return deltas