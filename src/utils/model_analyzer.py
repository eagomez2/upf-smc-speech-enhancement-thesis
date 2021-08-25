"""
Based on the  original work by:
Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""
import sys
import torch
import time
import platform
import torch.nn as nn
import numpy as np
from .custom_modules import CausalConv1d
from prettytable import PrettyTable
from functools import partial


class ModelAnalyzer:
    """ Class that provides the tools to analyze the complexity of a given
        model.

    custom_module_hooks_map (dict): Dictionary containing custom hook functions
        to be associated with custom nn.Modules.
    module_exceptions (list): List containing the modules that are not
        considered while computing the analysis results.
    verbose: If True, verbose mode is enabled.
    """
    def __init__(self, custom_module_hooks_map={}, 
                 module_exceptions=[], verbose=False):
        super().__init__()
        self.verbose = verbose

        # dictionary containing the hook maps per module
        self.module_hooks_map = {
                # linear layers
                nn.Linear: self._linear_flops_fn,

                # activation functions
                nn.ReLU: self._relu_flops_fn,
                nn.PReLU: self._relu_flops_fn,
                nn.ELU: self._relu_flops_fn,
                nn.LeakyReLU: self._relu_flops_fn,
                nn.ReLU6: self._relu_flops_fn,

                # convolutional layers
                nn.Conv1d: self._conv_flops_fn,
                nn.Conv2d: self._conv_flops_fn,
                nn.Conv3d: self._conv_flops_fn,

                # transposed convolutional layers
                nn.ConvTranspose1d: self._conv_flops_fn,
                nn.ConvTranspose2d: self._conv_flops_fn,
                nn.ConvTranspose3d: self._conv_flops_fn,
                   CausalConv1d: self._conv_flops_fn,

                # recurrent layers
                nn.RNN: self._rnn_flops_fn,
                nn.GRU: self._rnn_flops_fn,
                nn.LSTM: self._rnn_flops_fn,
                nn.RNNCell: self._rnn_cell_flops_fn,
                nn.GRUCell: self._rnn_cell_flops_fn,
                nn.LSTMCell: self._rnn_cell_flops_fn,
                nn.MultiheadAttention: self._multihead_attention_flops_fn,

                # pooling layers
                nn.MaxPool1d: self._pooling_flops_fn,
                nn.MaxPool2d: self._pooling_flops_fn,
                nn.MaxPool3d: self._pooling_flops_fn,
                nn.AvgPool1d: self._pooling_flops_fn,
                nn.AvgPool2d: self._pooling_flops_fn,
                nn.AvgPool3d: self._pooling_flops_fn,
                nn.AdaptiveMaxPool1d: self._pooling_flops_fn,
                nn.AdaptiveMaxPool2d: self._pooling_flops_fn,
                nn.AdaptiveMaxPool3d: self._pooling_flops_fn,
                nn.AdaptiveAvgPool1d: self._pooling_flops_fn,
                nn.AdaptiveAvgPool2d: self._pooling_flops_fn,
                nn.AdaptiveAvgPool3d: self._pooling_flops_fn,

                # batch normalization layers
                nn.BatchNorm1d: self._batch_norm_flops_fn,
                nn.BatchNorm2d: self._batch_norm_flops_fn,
                nn.BatchNorm3d: self._batch_norm_flops_fn,

                # upsample
                nn.Upsample: self._upsample_flops_fn,
                }
        self.custom_module_hooks_map = custom_module_hooks_map
        self.module_exceptions = module_exceptions

    @torch.no_grad()
    def analyze_complexity(self, model, input):
        """ Analyzes the model complexity and prints a table containing
        the amount of computed flops needed on a per-layer basis.

        Args:
            model (nn.Module): The model to be analyzed.
            input (torch.tensor): A tensor containing a test input to be fed
                to the model.
        """
        # perform sanity check
        self._perform_complexity_sanity_check(model)

        # reset flops counter of the model and all its modules
        self._reset_module_batch_counter(model)
        model.apply(self._reset_module_flops_counter)
        model.apply(self._reset_module_params_counter)
        model.eval()

        # start flops count
        self._add_batch_counter_hook_fn(model)

        # store different types of observed modules
        self.observed_module_types = set()

        # add flops counter hook fn
        model.apply(self._add_flops_counter_hook_fn)

        # perform forward pass
        model(input)

        # count flops and params
        total_flops = self._compute_mean_flops_cost(model)
        total_params = self._compute_trainable_params(model)

        # print results
        self._print_model_with_flops(model, total_flops, total_params)

        # teardown
        self._remove_batch_counter_hook_fn(model)
        model.apply(self._remove_batch_counter_hook_fn)
        model.apply(self._remove_params_counter)

        # delete observed modules after computations are completed
        del self.observed_module_types

    @torch.no_grad()
    def analyze_inference_speed(self, model, input, 
            batch_correction=1.0, iters_n=1000, drop_initial_n_iters=10):
        """ Computes the inference speed required on a single forward pass. 

        IMPORTANT: Some models cannot be evaluated on a single batch, therefore
        the batch_correction factor will divide the final result by the number
        of batches to obtain the speed of predicting a single input batch.

        Args:
            batch_correction (float): Factor to divide by the inference speed
                for models needing more than a single batch to return a prediction.
            iters_n (int): Number of iterations to be performed.
            drop_initial_n_iters (int): First iterations are known to take
                longer times, therefore this parameter controls how many
                iterations are discarded to perform the actual statistical
                aggregation of the results.
        """
        # batch
        model.eval()
        exec_times = []  # stores execution times

        for idx in range(iters_n):
            start = time.perf_counter()
            model(input)
            end = time.perf_counter()
            exec_times.append(end - start)

        # drops drop_initial_n_iters
        exec_times = np.array(exec_times)[drop_initial_n_iters:]

        # apply correction and turn into ms
        exec_times = exec_times * 1000 * batch_correction

        # mean exec_time
        min_exec_time = np.min(exec_times)
        max_exec_time = np.max(exec_times)
        mean_exec_time = np.mean(exec_times)
        stdev_exec_time = np.std(exec_times)
        perc20_exec_time = np.percentile(exec_times, 20)
        perc40_exec_time = np.percentile(exec_times, 40)
        median_exec_time = np.median(exec_times)
        perc60_exec_time = np.percentile(exec_times, 60)
        perc80_exec_time = np.percentile(exec_times, 80)
        
        summary_table = PrettyTable()
        summary_table.field_names = ["Stats", "Time"]
        summary_table.add_row(["Min", f"{min_exec_time:.3f}ms"]) 
        summary_table.add_row(["Max", f"{max_exec_time:.3f}ms"]) 
        summary_table.add_row(["Mean", f"{mean_exec_time:.3f}ms"])
        summary_table.add_row(["Stdev", f"{stdev_exec_time:.3f}ms"])
        summary_table.add_row(["20%", f"{perc20_exec_time:.3f}ms"])
        summary_table.add_row(["40%", f"{perc40_exec_time:.3f}ms"])
        summary_table.add_row(["Median", f"{median_exec_time:.3f}ms"])
        summary_table.add_row(["60%", f"{perc60_exec_time:.3f}ms"])
        summary_table.add_row(["80%", f"{perc80_exec_time:.3f}ms"])
        print(f"{model.__class__.__name__} inference speed summary")
        print(summary_table)
        print(f"System: {platform.system()} {platform.processor()}")

    def _reset_module_batch_counter(self, module):
        module.__batch_counter__ = 0

    def _reset_module_flops_counter(self, module):
        if self.is_supported_module(module):
            if hasattr(module, "__flops__"):
                print("Warning: Property __flops__ already defined in "
                     f"module {type(module).__name__} "
                     "complexity analysis can affect your code")
            module.__flops__ = 0

    def _reset_module_params_counter(self, module):
        # trainable parameters get calculated right away since there is
        # need for extra information
        if self.is_supported_module(module):
            if hasattr(module, "__prams__"):
                print("Warning: Property __params__ already defined in "
                     f"module {type(module).__name__} "
                     "complexity analysis can affect your code")

            params_n = sum(
                    p.numel() for p in module.parameters() if p.requires_grad)
            module.__params__ = params_n

    def _add_batch_counter_hook_fn(self, module):
        if hasattr(module, "__batch_counter_fn__"):
            return

        handle = module.register_forward_hook(self._batch_counter_hook_fn)
        module.__batch_counter_fn__ = handle

    def _remove_batch_counter_hook_fn(self, module):
        if hasattr(module, "__batch_counter_fn__"):
            del module.__batch_counter_fn__

    @staticmethod
    def _batch_counter_hook_fn(module, input, output):
        if len(input) > 0:
            # module can have multiple inputs, getting the first one
            input = input[0]
            batch_size = len(input)
        else:
            raise RuntimeError("No inputs found for a module {module.__name__}")

        module.__batch_counter__ += batch_size

    def _perform_complexity_sanity_check(self, model):
        if not isinstance(model, nn.Module):
            raise ValueError("The model should be an instance of nn.Module. "
                            f"Found model of type {type(model)}")

    def _add_flops_counter_hook_fn(self, module):
        # if is exception module
        if type(module) in self.module_exceptions:
            self.observed_module_types.add(type(module))

            if self.is_supported_module(module):
                module.__params__ = 0

        # regular module
        elif self.is_supported_module(module):
            if hasattr(module, "__flops_fn__"):
                return

            if type(module) in self.custom_module_hooks_map:
                handle = module.register_forward_hook(
                        self.custom_module_hooks_map[type(module)])
            else:
                handle = module.register_forward_hook(
                        self.module_hooks_map[type(module)])

            module.__flops_fn__ = handle
            self.observed_module_types.add(type(module))

        # module not found
        else:
            if (self.verbose and not type(module) in 
                    (nn.Sequential, nn.ModuleList)
                        and not type(module) in self.observed_module_types):
                print(f"Warning: module {type(module).__name__} treated as "
                       " zero-op")

            self.observed_module_types.add(type(module))

    def _remove_flops_counter_hook_fn(self, module):
        if self.is_supported_module(module):
            if hasattr(module, "__flops_fn__"):
                del module.__flops_fn__

    def _remove_params_counter(self, module):
        if self.is_supported_module(module):
            if hasattr(module, "__params__"):
                del module.__params__

    def is_supported_module(self, module):
        """ Checks if a module has a flops_count_hook_fn available in
        the module_hooks_map. """
        return (type(module) in self.module_hooks_map or 
            type(module) in self.custom_module_hooks_map)

        if type(module) in self.module_exceptions:
            # is model is in exception list, it is just added to the observed
            # model types
            self.observed_module_types.add(type(module))

            # if supported, params will not be counted
            if self.is_supported_module(module):
                module.__params__ = 0

        elif self.is_supported_module(module):
            if hasattr(module, "__flops_counter_fn__"):
                return
            if type(module) in self.custom_module_hooks_map:
                # register hook if found in custom_hooks_map
                handle = module.register_forward_hook(
                            self.custom_module_hooks_map[type(module)])
            else:
                handle = module.register_forward_hook(
                            self.modules_hooks_map[type(module)])
 
            # adds property with functions used to calculate flops
            module.__flops_fn__ = handle
            self.observed_module_types(type(module))
        else:
            if (self.verbose and not type(module) in  
                (nn.Sequential, nn.ModuleList) and 
                type(module) in self.observed_module_types):
                print(f"Module {type(module).__name__} treated as zero-op")

            self.observed_module_types.add(type(module))

    def _accumulate_flops(self, module):
        if self.is_supported_module(module):
            return module.__flops__
        else:
            total_flops = 0

            for module in module.children():
                total_flops += self._accumulate_flops(module)

            return total_flops

    def _accumulate_params(self, module):
        if self.is_supported_module(module):
            return module.__params__
        else:
            total_params = 0

            for module in module.children():
                total_params += self._accumulate_params(module)

            return total_params

    def _get_model_params(self, model):
        trainable_params = 0
        nontrainable_params = 0

        for param in model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                nontrainable_params += param.numel()

        return trainable_params, nontrainable_params

    # TODO: deprecate in favour of _get_model_params
    def _compute_trainable_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _compute_mean_flops_cost(self, model): 
        total_flops = self._accumulate_flops(model)
        total_flops_per_batch = total_flops / model.__batch_counter__
        return int(total_flops_per_batch)

    def _print_model_with_flops(self, model, total_flops, total_params,
                                precision=3, units=None):
        summary_table = PrettyTable()
        summary_table.field_names = [
                "#", "Layer", "Type", 
                "Parameters", "Parameters %", 
                "Flops", "Flops %"
                ]

        for idx, (name, module) in enumerate(model.named_modules()):
            if idx > 0: # to avoid printing the whole model which is module[0]
                if hasattr(module, "__flops__"):
                    module_flops = module.__flops__
                    module_flops_percentage = module_flops / total_flops * 100
                    module_flops_percentage = f"{module_flops_percentage:.2f}%"
                else:
                    module_flops = "-"
                    module_flops_percentage = "-"

                if hasattr(module, "__params__"):
                    module_params = module.__params__
                    module_params_percentage = module_params / total_params * 100
                    module_params_percentage = f"{module_params_percentage:.2f}%"
                else:
                    module_params = "-"
                    module_params_percentage = "-"

                summary_table.add_row([idx - 1, 
                    name, module.__class__.__name__, 
                    module_params, module_params_percentage, 
                    module_flops, module_flops_percentage])

        (true_trainable_params, 
         true_nontrainable_params) = self._get_model_params(model)

        true_total_params = true_trainable_params + true_nontrainable_params

        print(f"{model.__class__.__name__} complexity analysis summary")
        print(summary_table)
        print("{:<25}{}".format("Total params:", true_total_params))
        print("{:<25}{}".format("Trainable params:", true_trainable_params))
        print("{:<25}{}".format("Non-trainable params:", 
                                true_nontrainable_params))
        print("{:<25}{}".format("Total flops per batch:", total_flops))
    
    def _linear_flops_fn(self, module, input, output):
        # this layer is just multiply and add, therefore the number of FLOPs
        # directly correlate with the dimensions of the output
        # elements_count = output.numel()
        # module.__flops__ += int(elements_count)
        input = input[0]
        output_last_dim = output.shape[-1]
        bias_flops = output_last_dim if module.bias is not None else 0
        module.__flops__ += int(
                np.prod(input.shape) * output_last_dim + bias_flops
                )

    def _rnn_flops(self, flops, module, w_ih, w_hh, input_size):
        # matrix matrix mult input hidden and internal hidden states
        flops += w_ih.shape[0] * w_ih.shape[1]

        # matrix matrix mult hidden hidden and internal state
        flops += w_hh.shape[0] * w_hh.shape[1]

        if isinstance(module, (nn.RNN, nn.RNNCell)):
            # add both operations
            flops += module.hidden_size
        elif isinstance(module, (nn.GRU, nn.GRUCell)):
            # hadamard product
            flops += module.hidden_size

            # adding operations from both states
            flops += module.hidden_size * 3

            # last two hadamard product and add
            flops += module.hidden_size * 3
        elif isinstance(module, (nn.LSTM, nn.LSTMCell)):
            # adding operations from both states
            flops += module.hidden_size * 4

            # two hadamard products and add for cell state
            flops += module.hidden_size * 3

            # final hadamard
            flops += module.hidden_size * 3
        
        return flops

    def _rnn_cell_flops_fn(self, module, input, output):
        flops = 0
        input_ = input[0]
        batch_size = input_.shape[0]
        w_ih = module.__getattr__("weight_ih")
        w_hh = module.__getattr__("weight_hh")
        input_size = input_.shape[1]
        flops = self._rnn_flops(flops, module, w_ih, w_hh, input_size)

        if module.bias:
            b_ih = module.__getattr__("bias_ih")
            b_hh = module.__getattr__("bias_hh")
            flops += b_ih.shape[0] + b_hh.shape[0]

        flops *= batch_size
        module.__flops__ += int(flops)

    def _rnn_flops_fn(self, module, input, output):
        """ Takes into account batch goes at first position, contrary to pytorch
        common rule (but actually it does not matter).
        If sigmoid and tanh are made hard, only a comparison flops should be
        accurate. <-- Not sure what the authors mean with this last sentence.
        """
        # input is a tuple containing a sequence to process and (optinally)
        # hidden and cell states
        flops = 0
        input_ = input[0]
        batch_size = input_.shape[0]
        seq_len = input_.shape[1]
        layers_n = module.num_layers

        for idx in range(layers_n):
            w_ih = module.__getattr__(f"weight_ih_l{idx}")
            w_hh = module.__getattr__(f"weight_hh_l{idx}")

            input_size = module.input_size if idx == 0 else module.hidden_size
            flops = self._rnn_flops(flops, module, w_ih, w_hh, input_size)

            if module.bias:
                b_ih = module.__getattr__(f"bias_ih_l{idx}")
                b_hh = module.__getattr__(f"bias_hh_l{idx}")
                flops += b_ih.shape[0] + b_hh.shape[0]

        flops *= batch_size
        flops *= seq_len

        if module.bidirectional:
            flops *= 2

        module.__flops__ += int(flops)

    def _multihead_attention_flops_fn(self, module, input, output):
        flops = 0
        q, k, v = input
        batch_size = q.shape[1]

        heads_n = module.num_heads
        embed_dim = module.embed_dim
        k_dim = module.kdim
        v_dim = module.vdim

        if k_dim is None:
            k_dim = embed_dim
    
        if v_dim is None:
            v_dim = embed_dim

        # initial projections
        flops = q.shape[0] * q.shape[2] * embed_dim + \
                k.shape[0] * k.shape[2] * k_dim + \
                v.shape[0] * v.shape[2] * v_dim

        if module.in_proj_bias is not None:
            flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

        # attention heads: scale, matmul, softmax, matmul
        # NOTE: not sure why the author wrote it like this since it can be
        # reduced
        head_dim = embed_dim // heads_n
        head_flops = q.shape[0] * head_dim + \
          head_dim * q.shape[0] * k.shape[0] + \
                     q.shape[0] * k.shape[0] + \
                     q.shape[0] * k.shape[0] * head_dim

        flops += heads_n * head_flops

    def _conv_flops_fn(self, module, input, output):
        # could have multiple inputs, getting the first one
        input_ = input[0]
        batch_size = input_.shape[0]
        output_dims = list(output.shape[2:])
        kernel_dims = list(module.kernel_size)
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * \
                in_channels * filters_per_channel

        active_elements_count = batch_size * int(np.prod(output_dims))
        total_conv_flops = conv_per_position_flops * active_elements_count
        bias_flops = 0

        if module.bias is not None:
            bias_flops = out_channels * active_elements_count

        total_flops = total_conv_flops + bias_flops
        module.__flops__ += int(total_flops)

    def _relu_flops_fn(self, module, input, output):
        elements_count = output.numel()
        module.__flops__ += int(elements_count)

    def _pooling_flops_fn(self, module, input, output):
        input_ = input[0]
        module.__flops__ += int(np.prod(input_.shape))

    def _batch_norm_flops_fn(self, module, input, output):
        input_ = input[0]

        flops = np.prod(input_.shape)

        if module.affine:
            flops *= 2

        module.__flops__ += int(flops)

    def _upsample_flops_fn(self, module, input, output):
        output_size = output[0]
        batch_size = output_size.shape[0]
        output_elements_count = batch_size

        for val in output_size.shape[1:]:
            output_elements_count *= val

        module.__flops__ += int(output_elements_count)
