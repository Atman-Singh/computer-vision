from datasets import DatasetDict
import torch
import torch.nn.functional as F
from torch.func import vmap

class ConvolutionalNeuralNetwork:

    def __init__(self, width: int, height: int, ds: DatasetDict):

        if torch.cuda.is_available():
            print("CUDA is available. PyTorch can use the GPU.")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version built with PyTorch: {torch.version.cuda}")
        else:
            print("CUDA is not available. PyTorch will use the CPU.")
        
        self.ds = ds
        labels = set()
        for l in ds['train']['label']:
            labels.add(l)
        self.output_range = len(labels)
        
        # convert pngs to tensors containing pixel values
        rows = len(ds['train'])
        width = width
        height = height
        if width != height:
            print('TODO: support for non-square images')

        self.images = torch.empty((rows, width, height), dtype=torch.int64)
        for i, row in enumerate(ds['train'].select(range(rows))):
            self.images[i] = torch.reshape(torch.tensor(list(row['image'].getdata())), (width, height))

        # define kernels
        torch.manual_seed(3500)
        min_fence = -0.2
        max_fence = 0.2

        # initialize kernels with random values and transform them to range [-0.2, 0.2)
        self.kernel_l1 = torch.rand(2, 1, 5, 5) * (max_fence - min_fence) + min_fence
        self.kernel_l2 = torch.rand(4, 1, 3, 3) * (max_fence - min_fence) + min_fence

        # initialize biases
        self.bias_l1 = torch.rand(2, 1) * (max_fence - min_fence) + min_fence
        self.bias_l2 = torch.rand(4, 1) * (max_fence - min_fence) + min_fence

        print(f'{self.kernel_l1.shape}\n{self.kernel_l2.shape}\n{self.bias_l1.shape}\n{self.bias_l2.shape}')

        kernel_stacks = [self.kernel_l1, self.kernel_l2]
        biases = [self.bias_l1, self.bias_l2]
        self.images_processed = 3

        activation_functions = ['relu', 'relu']
        input_x = self.images[:self.images_processed]
        output_conv = self.forward(input_x, kernel_stacks, None, biases, activation_functions, (2,2))
        deltas = self.backward(output_conv, kernel_stacks, input_x, ds, self.output_range)
        print('Finished CNN initialization.')

    # helper function to divide out rows from first dimension of tensor
    def _seperate(self, tensor: torch.Tensor, i: int):
        shape = tensor.shape
        if len(shape) > 1:
            return tensor.reshape(i, shape[0]//i, *shape[1:])
        return tensor.reshape(i, shape[0]//i)

    # computes convolution between kernel and matrix
    def _traverse_matrix(self, matrix: torch.Tensor, kernel: torch.Tensor, step: int) -> torch.Tensor:
        if len(kernel.shape) == 2:
            print('warning: kernel rank is 2, adjusting shape')
            kernel = kernel.unsqueeze(0)
        if len(kernel.shape) != 3:
            raise Exception(f"kernel has a rank of {len(kernel.shape)}")
        
        width, height = matrix.shape[1] - kernel.shape[-2] + 1, matrix.shape[0] - kernel.shape[-1] + 1

        # create tensor of all windows kernel will slide over and repeat by number of kernels
        matrix_unfolded = (
            matrix
                .unfold(0, width, step)
                .unfold(1, height, step)
                .permute(2,3,0,1)
        )

        kernel = kernel.unsqueeze(1).unsqueeze(1)
        unfolded_sum = kernel * matrix_unfolded
        return unfolded_sum.sum(dim=(-1,-2)).sum(0)

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
            pool_size: tuple
            ) -> list:
        MS_PER_MATRIX = 568.25
        print(f"ETA: {round(len(data) * (MS_PER_MATRIX / 1000 / 60), 2)} minutes")
        
        if not (isinstance(data, torch.Tensor) or isinstance(data, list)):
            raise Exception(f"Data is not of type 'list', inputted type is: {type(data)}")
        if not isinstance(kernel_stacks, list):
            raise Exception(f"Kernels is not of type 'list', inputted type is: {type(kernel_stacks)}")
        if len(kernel_stacks) > len(biases):
            raise Exception(f'number of kernel stacks {len(kernel_stacks)} greater than number of biases ({len(biases)})')
        if len(kernel_stacks) != len(activation_functions):
            print(f"Number of kernel stacks inputted ({len(kernel_stacks)}) does " +
                f"not equal number of activation functions inputted " +
                f"({len(activation_functions)})")

        images_processed = len(data)
        for i, kernel_stack in enumerate(kernel_stacks):
            kernel_stacks[i] = kernel_stack.unsqueeze(0).expand(images_processed, *kernel_stack.shape)
            kernel_stacks[i] = kernel_stacks[i].reshape(-1, *kernel_stacks[i].shape[-3:])

        for i, bias in enumerate(biases):
            if i < len(kernel_stacks):
                biases[i] = (
                    bias
                    .repeat(images_processed,1)
                    .unsqueeze(-1)
                )
        
        # define working input
        input_stack = data
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
            print(f'scale: {kernel_stack.shape[0] // input_stack.shape[0]}')
            input_stack = input_stack.repeat_interleave(kernel_stack.shape[0] // input_stack.shape[0], dim=0)
            print(f'input: {input_stack.shape}\nkernel: {kernel_stack.shape}')
            
            # take convolution
            convolution = vmap(self._traverse_matrix, in_dims=(0,0,None))(input_stack, kernel_stack, 1) 
            print(convolution.shape)
            convolution = convolution + biases[i]
            # take activation
            activation = func(convolution)
            # pool activation
            pooled = vmap(self._max_pool, in_dims=(0,None))(activation, pool_size)

            # update output
            output[i] = (convolution, activation, pooled)
            # update working input
            input_stack = pooled
            print(f'\nconvolution: {convolution.shape}\nactivation: {activation.shape}\npooled: {pooled.shape}\n')

        # flattening
        flattened = (
            self._seperate(pooled, images_processed)
            .flatten(1)
            .unsqueeze(-1)
        )

        # initialize weights and biases for fully connected layer if not already
        if weights is None and len(biases) == len(kernel_stacks):
            torch.manual_seed(3500)
            min_fence = -0.2
            max_fence = 0.2
            
            weights = torch.rand(self.output_range, flattened.shape[1]) * (max_fence - min_fence) + min_fence
            biases.append(torch.rand(self.output_range, 1) * (max_fence - min_fence) + min_fence)

        # compute fully connected layer output
        eps = 1e-7
        fc_output = (weights @ flattened.T + biases[-1]).squeeze(0)
        func = torch.nn.Sigmoid()
        activation = torch.clamp(
            func(fc_output), 
            eps, 
            1-eps
        )
        output[-1] = (flattened, weights, biases, activation)
        
        print(f'flattened: {flattened.shape}\nweights: {weights.shape}\nbiases: {biases[-1].shape}\nactivation: {activation.shape}]\n')
        return output

    # binary cross-entropy loss function
    def _compute_loss(self, a: torch.Tensor, y: torch.Tensor):
        return y * torch.log(a) + (1 - y) * torch.log(1 - a) 

    # function to compute cost of output for all images
    def _compute_cost(self, output: torch.Tensor, dataset: DatasetDict):
        cost = torch.zeros(self.output_range)
        images_computed = output.shape[-1]
        for i in range(output.shape[1]):
            label = dataset['train'][i]['label']
            eps = 1e-7
            output_v = torch.clamp(output[:, i], eps, 1-eps)
            expected_v = torch.zeros(self.output_range)
            expected_v[label] = 1
            
            cost += self._compute_loss(output_v, expected_v)
        cost /= images_computed
        return -cost.unsqueeze(1)

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
            dataset: DatasetDict, 
            output_range: int
            ):
        # compute loss
        activation_fc = forward_output[-1][3]
        cost = self._compute_cost(activation_fc, dataset)
        images_computed = activation_fc.shape[-1]
        
        # compute fully connected layer deltas
        biases_fc_d = None
        flattened, weights, biases_fc = forward_output[-1][0], forward_output[-1][1], forward_output[-1][2]

        labels = torch.tensor(self.ds['train']['label'], dtype=torch.long)
        expected = torch.zeros(output_range, images_computed)
        expected[labels[:images_computed], torch.arange(images_computed)] = 1
        
        output_fc_d = activation_fc - expected #dZ, kept seperate per image
        weights_d = output_fc_d @ flattened.squeeze() / images_computed #dW, averaged
        biases_fc_d = torch.sum(output_fc_d, dim=1) / images_computed #dB, averaged
        flattened_d = weights.T @ output_fc_d #dF, kept seperate
        
        print(f'fully connected layer\noutput deltas: {output_fc_d.shape}\nweights deltas: {weights_d.shape}')
        print(f'biases: {biases_fc_d.shape}\nflattened deltas: {flattened_d.shape}\n')
        
        # compute convolutional pass deltas, iterate backwards starting from last convolutional layer
        deltas = [None for _ in range(len(kernel_stacks))]
        for i, cl_output in reversed(list(enumerate(forward_output[:-1]))):
            print(f'layer {i}')
            convolution, activation, pooled = forward_output[i]
            
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
                    .flatten(0,1)
                    .rot90(2,(-2,-1))
                )

                pooled_d_temp = (
                    self._seperate(
                        vmap(self._traverse_matrix, in_dims=(0,0,None))(prev_convolution_d_padded, rotated_kernel, 1),
                        self.images_processed
                    )
                )
                
                channel_f = prev_kernel.shape[0] // kernel.shape[0]
                s = pooled_d_temp.shape
                pooled_d = (
                    pooled_d_temp
                        .reshape(s[0], channel_f, s[1] // channel_f, *s[2:])
                        .sum(2)
                        .flatten(0,1)
                )

            # compute deltas for layer 2 activations
            activation_d = self._seperate(
                vmap(self._max_pool_d, in_dims=(0, 0, None))(activation, pooled_d, (2,2)),
                self.images_processed
            )

            # compute activation deltas for convolutionns
            convolution_d_activation = self._seperate(self._relu_d(convolution), images_computed)

            # compute cost deltas for convolutions
            convolution_d = (convolution_d_activation * activation_d).flatten(0,1)

            if i == 0:
                # if at the beginning convolutional layer, input tensor is the input to the convolutional neuron
                input_t = input_x.repeat_interleave(2, dim=0) 
            else:
                # next pooled matrix is input to convolutional neuron
                input_t = forward_output[i-1][2].repeat_interleave(2, dim=0) 

            kernel_d = self._seperate(
                vmap(self._traverse_matrix, in_dims=(0,0,None))(input_t, convolution_d, 1),
                self.images_processed
            )

            # compute deltas for layer 2 biases, averaged and used to update biases
            convolution_d_seperated = self._seperate(convolution_d, self.images_processed)
            bias_d = convolution_d_seperated.sum((0,2,3)) / self.images_processed
            
            print(f'pooled deltas: {pooled_d.shape}\nactivation deltas: {activation_d.shape}\nconvolution deltas: {convolution_d.shape}')
            print(f'kernel deltas: {kernel_d.shape}\nbias deltas: {bias_d.shape}\n')
            deltas[i] = (convolution_d, kernel_d, bias_d)
        return deltas