#!/usr/bin/env python3

import collections
import numpy as np
from scipy.stats import ks_1samp, kstest
import scipy.stats
from pathlib import Path
import argparse
import os
import sys
import gnuplotlib as gp
import ctypes

llama_base = '.'
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / 'gguf-py'))
    llama_base = str(Path(__file__).parent)

from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType

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
    libllama = ctypes.CDLL(os.path.join(llama_base, 'libllama.so'))
except OSError:
    try:
        libllama = ctypes.CDLL(os.path.join(llama_base, 'libllama.dll'))
    except OSError:
        libllama = ctypes.CDLL(os.path.join(llama_base, 'libllama.dylib'))

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

        for tensor in reader.tensors:
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
    else:
        # remove test.gguf
        try:
            os.remove('test.gguf')
        except FileNotFoundError:
            pass

        apply(models = models,
              scheme = Experiment1QuantScheme(),
              trained = None,
              output_gguf_filepath = 'test.gguf')

def get_tensor(reader, name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return dequantize(tensor)
    raise ValueError(f"Tensor {name} not found in model")

class Experiment1QuantScheme:
    def __init__(self):
        pass

    def name(self):
        return "literally just passthrough"

    def train(self, tensors_by_name):
        return None

    def apply(self, trained, tensors_by_name):
        for name, tensor in tensors_by_name.items():
            yield name, tensor[0]

def apply(models, scheme, trained, output_gguf_filepath):
    # Copy metadata from the first model
    reader = GGUFReader(models[0])
    host_endian, file_endian = get_file_host_endian(reader)

    arch = None
    name = None
    for idx, field in enumerate(reader.fields.values()):
        if field.name == 'general.architecture':
            assert field.types[-1] == GGUFValueType.STRING
            arch = str(bytes(field.parts[-1]), encoding='utf8')
        elif field.name == 'general.name':
            assert field.types[-1] == GGUFValueType.STRING
            name = str(bytes(field.parts[-1]), encoding='utf8')

        if arch and name:
            break

    if not arch:
        raise ValueError("Architecture not found (general.architecture)")

    if not name:
        raise ValueError("Name not found (general.name)")

    writer = GGUFWriter(output_gguf_filepath, arch, use_temp_file=False)
    writer.add_name(name)

    writer.add_string('general.created-by', 'octopus-merge')

    # Copy all the fields (metadata) from the first model
    for idx, field in enumerate(reader.fields.values()):
        if field.name.startswith('GGUF.'):
            continue
        if field.name == 'general.architecture':
            continue
        if field.name == 'general.name':
            continue

        # No easy way to just write what you read? well whatever
        # the set of types is typically fairly short
        if field.types == [GGUFValueType.UINT32]:
            writer.add_uint32(field.name, field.parts[-1][0])
        elif field.types == [GGUFValueType.UINT64]:
            writer.add_uint64(field.name, field.parts[-1][0])
        elif field.types == [GGUFValueType.STRING]:
            writer.add_string(field.name, str(bytes(field.parts[-1]), encoding='utf8'))
        elif field.types == [GGUFValueType.FLOAT32]:
            writer.add_float32(field.name, field.parts[-1][0])
        elif field.types == [GGUFValueType.BOOL]:
            writer.add_bool(field.name, field.parts[-1][0])
        elif field.types == [GGUFValueType.ARRAY, GGUFValueType.STRING]:
            writer.add_array(field.name, [str(bytes(field.parts[idx]), encoding='utf8') for idx in field.data])
        elif field.types == [GGUFValueType.ARRAY, GGUFValueType.FLOAT32]:
            writer.add_array(field.name, [pv for idx in field.data for pv in field.parts[idx].tolist()])
        elif field.types == [GGUFValueType.ARRAY, GGUFValueType.INT32]:
            writer.add_array(field.name, [pv for idx in field.data for pv in field.parts[idx].tolist()])
        else:
            raise ValueError(f"Unsupported field type {field.types}")

    del reader

    # and then all the tensors
    tensors_by_name = collections.defaultdict(lambda: [])
    model_readers = []
    for model in models:
        reader = GGUFReader(model)
        model_readers.append(reader)

        for tensor in reader.tensors:
            tensors_by_name[tensor.name].append(tensor)

    already_written = set()
    for name, tensor in scheme.apply(trained, tensors_by_name):
        if name in already_written:
            raise ValueError(f"Tensor {name} already written")

        # reverse shape
        # why does the shape have to be reversed? I don't know
        # but that's how I got it to work
        shape = list(tensor.shape)[::-1]

        writer.add_tensor(name=name,
                          tensor=tensor.data,
                          raw_shape=tuple(shape),
                          raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    writer.close()


def train(target_model, models, scheme):
    models = set(models) - {target_model}

    #if not models:
    #    print("Error: No models to train with.")
    #    sys.exit(1)

    tensors = set()
    target_reader = GGUFReader(target_model, 'r')
    for tensor in target_reader.tensors:
        tensors.add(tensor.name)
    model_readers = [GGUFReader(model, 'r') for model in models]

    tensors_by_name = {}
    for tensor in tensors:
        target_tensor = get_tensor(target_reader, tensor)
        model_tensors = []
        for model_reader in model_readers:
            model_tensor = get_tensor(model_reader, tensor)
            model_tensors.append(model_tensor)
        tensors_by_name[tensor] = (target_tensor, model_tensors)

    result = scheme.train(tensors_by_name)

    print("Training complete: ", result)

if __name__ == '__main__':
    main()
