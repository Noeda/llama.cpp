#!/usr/bin/env python3

import collections
import numpy as np
from scipy.stats import ks_1samp, kstest
import scipy.stats
from gguf import GGUFReader, GGUFValueType
from gguf.constants import GGMLQuantizationType
import argparse
import os
import sys
import gnuplotlib as gp
import ctypes

# Stupid shims to call some of the dequantize functions from libllama
# Added just the ones I've been using for testing.
# void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
# void quantize_row_q2_K_reference(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int k);
# void quantize_row_q4_K_reference(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int k);
# void quantize_row_q8_0_reference(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);
# void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
# void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
# void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
# void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

try:
    libllama = ctypes.CDLL('libllama.so')
except OSError:
    try:
        libllama = ctypes.CDLL('libllama.dll')
    except OSError:
        libllama = ctypes.CDLL('libllama.dylib')

# x = source data
# y = destination data
# k = number of elements in x

def quantize_row_q5_K(x, y, k):
    libllama.quantize_row_q5_K(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

def quantize_row_q2_K_reference(x, y, k):
    libllama.quantize_row_q2_K_reference(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

def quantize_row_q4_K_reference(x, y, k):
    libllama.quantize_row_q4_K_reference(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

def quantize_row_q8_0_reference(x, y, k):
    libllama.quantize_row_q8_0_reference(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

def dequantize_row_q8_0(x, y, k):
    libllama.dequantize_row_q8_0(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

def dequantize_row_q2_K(x, y, k):
    libllama.dequantize_row_q2_K(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

def dequantize_row_q5_K(x, y, k):
    libllama.dequantize_row_q5_K(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

def dequantize_row_q4_K(x, y, k):
    libllama.dequantize_row_q4_K(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

def dequantize(tensor):
    if tensor.data.dtype == np.float32:
        return tensor.data.reshape(tensor.shape)
    if tensor.data.dtype == np.float16:
        return tensor.data.astype(np.float32).reshape(tensor.shape)

    num_elems = tensor.n_elements.item()
    # numpy for dequanted data
    target_data = np.zeros(num_elems, dtype=np.float32)
    if tensor.tensor_type == GGMLQuantizationType.Q8_0:
        dequantize_row_q8_0(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q2_K:
        dequantize_row_q2_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q5_K:
        dequantize_row_q5_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q4_K:
        dequantize_row_q4_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)

    raise ValueError(f"Unsupported tensor type {tensor.tensor_type}")

def arg_parser():
    parser = argparse.ArgumentParser(description='Octopus merge thing')
    parser.add_argument('model', type=str, help='GGUF model to use. Specify more than one, or this script is not really useful.', nargs='+')
    parser.add_argument('--train', help='Train an octopus quant model. The model with highest precision will be used as the target.', default=False, action='store_true')
    return parser

def get_file_host_endian(reader: GGUFReader) -> tuple[str, str]:
    host_endian = 'LITTLE' if np.uint32(1) == np.uint32(1).newbyteorder("<") else 'BIG'
    if reader.byte_order == 'S':
        file_endian = 'BIG' if host_endian == 'LITTLE' else 'LITTLE'
    else:
        file_endian = host_endian
    return (host_endian, file_endian)

def main():
    args = arg_parser().parse_args()

    models = args.model
    if not models:
        print("Error: No models specified")
        sys.exit(1)

    for model in models:
        if not os.path.isfile(model):
            print(f"Error: {model} does not exist")
            sys.exit(1)

    tensor_counts = collections.defaultdict(lambda: 0)

    # Assume: more bytes = higher precision
    model_total_bytes = collections.defaultdict(lambda: 0)

    for model in models:
        reader = GGUFReader(model, 'r')
        host_endian, file_endian = get_file_host_endian(reader)

        for n, tensor in enumerate(reader.tensors, 1):
            tensor_counts[tensor.name] += 1
            model_total_bytes[model] += tensor.n_bytes

    expected_count = None
    highest_precision_model = None
    highest_precision_model_bytes = 0

    for model in models:
        if highest_precision_model is None or model_total_bytes[model] > highest_precision_model_bytes:
            highest_precision_model = model
            highest_precision_model_bytes = model_total_bytes[model]

    for name, count in tensor_counts.items():
        if expected_count is None:
            expected_count = count
            continue

        if count != expected_count:
            print(f"Error: {name} has {count} tensors, expected {expected_count}")
            sys.exit(1)

        print(f"{name} has {count} tensors across models - good")

    print(f'Highest precision model is {highest_precision_model} at {highest_precision_model_bytes} bytes.')

    if args.train:
        train(target_model = highest_precision_model,
              models = models)

def get_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return dequantize(tensor)
    raise ValueError(f"Tensor {name} not found in model")

def train(target_model, models):
    models = set(models) - {target_model}

    #if not models:
    #    print("Error: No models to train with.")
    #    sys.exit(1)

    tensors = set()
    target_reader = GGUFReader(target_model, 'r')
    for tensor in target_reader.tensors:
        tensors.add(tensor.name)
    model_readers = [GGUFReader(model, 'r') for model in models]

    for tensor in tensors:
        target_tensor = get_tensor(target_reader, tensor)
        for model_reader in model_readers:
            model_tensor = get_tensor(model_reader, tensor)
            diff = target_tensor - model_tensor
            diff = diff.flatten()

            # TODO: actually do something here.

    print('Training done')

if __name__ == '__main__':
    main()
