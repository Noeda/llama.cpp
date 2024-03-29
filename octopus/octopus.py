#!/usr/bin/env python3

import faulthandler
faulthandler.enable()

import json
import concurrent.futures as cf
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
import tqdm

llama_base = '.'
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent / 'gguf-py'))
    llama_base = str(Path(__file__).parent)

from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType

# Stupid shims to call some of the dequantize functions from libllama
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

def quantize_row_q3_K(x, y, k):
    libllama.quantize_row_q3_K(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

def quantize_row_q6_K(x, y, k):
    libllama.quantize_row_q6_K(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y, k)

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

def dequantize_row_q3_K(x, y, k):
    libllama.dequantize_row_q3_K(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

def dequantize_row_q6_K(x, y, k):
    libllama.dequantize_row_q6_K(x, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k)

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
    if tensor.tensor_type == GGMLQuantizationType.Q3_K:
        dequantize_row_q3_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q5_K:
        dequantize_row_q5_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q4_K:
        dequantize_row_q4_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
        return target_data.reshape(tensor.shape)
    if tensor.tensor_type == GGMLQuantizationType.Q6_K:
        dequantize_row_q6_K(tensor.data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), target_data, num_elems)
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
              scheme = Experiment2QuantScheme(),
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
            return tensor
    raise ValueError(f"Tensor {name} not found in model")

class Experiment1QuantScheme:
    def __init__(self):
        pass

    def name(self):
        return "literally just passthrough for whatever is the first source model"

    def train(self, tensors_by_name):
        return None

    def apply(self, trained, tensors_by_name):
        for name, tensor in tensors_by_name.items():
            yield name, tensor[0]

from ephemeral_cache import ECache
dcache = ECache()

def evaluate_quant2_base_model(scheme, tensors_by_name, model_idx_to_use, model_name, max_workers=1):
    global dcache

    with cf.ThreadPoolExecutor(max_workers) as executor:
        def handler(tensor_idx, original_target, source_tensors):
            def compute_target(key):
                if key[0] == 'target':
                    return dequantize(original_target)
                assert False

            def action(target):
                msesum = np.zeros(1, dtype=np.float64)
                total_elems = 0

                if model_idx_to_use == len(source_tensors):
                    # deliberately use the target model, and
                    # dequantize it twice (smoke testing)
                    weight_vec = dequantize(original_target)
                else:
                    weight_vec = dequantize(source_tensors[model_idx_to_use])

                msesum[0] += np.sum((weight_vec.astype(np.float64) - target.astype(np.float64)) ** 2)
                total_elems += target.size

                msesum = msesum.item()

                return msesum, total_elems

            return dcache.with_value(('target', tensor_idx), compute_target, action)

        text = f'Quant MSE evaluator base model: {model_name}'

        futs = []
        for tensor_idx, (name, (target, source_tensors)) in enumerate(tensors_by_name.items()):
            futs.append(executor.submit(handler, tensor_idx, target, source_tensors))

        msesum = np.zeros(1, dtype=np.float64)
        total_elems = 0

        tq = tqdm.tqdm(cf.as_completed(futs), total=len(futs), desc=text)
        for fut in tq:
            try:
                sub_msesum, sub_total_elems = fut.result()
            except Exception as e:
                # Eagerly print the exception, and then re-raise it
                sys.stderr.write(f'Error in {model_name}: {e}\n')
                raise
            msesum[0] += sub_msesum
            total_elems += sub_total_elems

            tq.set_postfix({f'RMSE': np.sqrt(msesum[0] / total_elems)})

    return model_name, np.sqrt(msesum[0] / total_elems)

def evaluate_quant2(scheme, tensor_idxs_for_this_round, quant_to_parameter_dict, tensors_by_name, candidate, idx, max_workers=1):
    global dcache

    score = 0

    bias = np.float32(0.0)
    weight_nps = np.zeros(1, dtype=np.float32)

    random_weights_baseline_calculation = False

    if candidate is not None:
        bias = np.float32(candidate[-1])
        weight_nps = [np.float32(x) for x in candidate[:-1]]
    else:
        random_weights_baseline_calculation = True

    mses = []
    n_tensors = len(tensors_by_name)
    text = 'Quant MSE evaluator idx=#{position}'.format(position=idx)

    if tensor_idxs_for_this_round is not None:
        tensor_idxs_for_this_round = set(tensor_idxs_for_this_round)

    msesum = np.zeros(1, dtype=np.float64)
    total_elems = 0

    with cf.ThreadPoolExecutor(max_workers) as executor:
        def handler(tensor_idx, target, source_tensors):
            if tensor_idx is not None and tensor_idxs_for_this_round is not None and tensor_idx not in tensor_idxs_for_this_round:
                return 0, 0

            msesum = np.zeros(1, dtype=np.float64)
            total_elems = 0

            def compute_target(key):
                assert key[1] == tensor_idx
                if key[0] == 'target':
                    return dequantize(target)
                if key[0] == 'source':
                    return (dequantize(source_tensors[key[2]]), source_tensors[key[2]].tensor_type)
                assert False

            def action(val):
                target = val[0]
                source_tensors = val[1:]

                nonlocal msesum, total_elems

                if random_weights_baseline_calculation:
                    mean = np.mean(target)
                    std = np.std(target)

                    rand_noise = np.random.normal(mean, std, target.shape)
                    msesum[0] += np.sum((rand_noise.astype(np.float64) - target.astype(np.float64)) ** 2)
                    total_elems += target.size
                else:
                    result = bias
                    for source, tensor_type in source_tensors:
                        weight = weight_nps[quant_to_parameter_dict[tensor_type]]
                        if weight != 0.0:
                            result += weight * source

                    msesum[0] += np.sum((result.astype(np.float64) - target.astype(np.float64)) ** 2)
                    total_elems += target.size

                return msesum.item(), total_elems

            keys = []
            keys.append(('target', tensor_idx))
            for idx in range(len(source_tensors)):
                keys.append(('source', tensor_idx, idx))

            return dcache.with_values(keys, compute_target, action)


        it = tensors_by_name.items()
        if max_workers == 1:
            tq = tqdm.tqdm(tensors_by_name.items(), total=n_tensors, desc=text)
            for tensor_idx, (name, (target, source_tensors)) in enumerate(tq):
                sub_msesum, sub_total_elems = handler(tensor_idx, target, source_tensors)
                msesum[0] += sub_msesum
                total_elems += sub_total_elems

                if total_elems > 0:
                    tq.set_postfix({f'RMSE {idx}': np.sqrt(msesum[0] / total_elems)})
        else:
            futures = [executor.submit(handler, tensor_idx, target, source_tensors) for tensor_idx, (name, (target, source_tensors)) in enumerate(it)]
            for future in tqdm.tqdm(cf.as_completed(futures), total=n_tensors, desc=text):
                sub_msesum, sub_total_elems = future.result()
                msesum[0] += sub_msesum
                total_elems += sub_total_elems


    score = np.sqrt(msesum[0] / total_elems) # RMSE

    return (idx, score)

class Experiment2QuantScheme:
    def __init__(self):
        pass

    def name(self):
        return "linear combination of all the quants, optimized for minimum MSE"

    def train(self, tensors_by_name, target_name, model_idx_to_model_name):
        import cma
        import sqlite3

        print('Updating results in train_results.sqlite3')

        conn = sqlite3.connect('train_results.sqlite3')
        cursor = conn.cursor()

        cursor.execute('CREATE TABLE IF NOT EXISTS train_results (now DATETIME DEFAULT CURRENT_TIMESTAMP, model TEXT NOT NULL, rmse REAL NOT NULL, random_baseline_adjusted_rmse_score REAL NOT NULL, balanced_baseline_adjusted_rmse_score REAL NOT NULL, epoch INTEGER NOT NULL, cmaes_sigma REAL)')
        cursor.execute('CREATE TABLE IF NOT EXISTS random_baseline_score ( random_baseline_score REAL NOT NULL )')
        cursor.execute('CREATE TABLE IF NOT EXISTS balanced_baseline_score ( balanced_baseline_score REAL NOT NULL )')
        cursor.execute('CREATE TABLE IF NOT EXISTS baseline_scores_by_source_model ( source_model TEXT NOT NULL, baseline_score REAL NOT NULL )')
        conn.commit()

        # get the baseline score, if it exists
        random_baseline_score = None
        for row in cursor.execute('SELECT random_baseline_score FROM random_baseline_score LIMIT 1'):
            random_baseline_score = row[0]
            print('Re-using random baseline score:', random_baseline_score)
            break

        if random_baseline_score is None:
            # baseline score (random; computed if we pass None for a candidate)
            _, random_baseline_score = evaluate_quant2(self, None, None, tensors_by_name, None, None, max_workers=16)

            cursor.execute('INSERT INTO random_baseline_score (random_baseline_score) VALUES (?)', (random_baseline_score,))
            conn.commit()

        # every distinct type we see in the source tensors gets a weight
        # parameter.
        source_types = set()
        n_tensors = 0
        for name, (target, tensors) in tensors_by_name.items():
            n_tensors = len(tensors)
            for tensor in tensors:
                source_types.add(tensor.tensor_type)

        assert len(source_types) > 0
        n_source_models = len(source_types)

        # simple map from type to Nth parameter
        # the last parameter is for bias.
        source_type_to_index = dict(zip(source_types, range(n_source_models)))

        # initial candidate; which has bias 0 and just perfectly averages all the
        # sources.
        src_vec = (n_source_models + 1) * [1.0/n_tensors]
        src_vec[-1] = 0.0

        with cf.ThreadPoolExecutor(max_workers=16) as executor:
            futs = []

            check_models = [(idx, name) for idx, name in model_idx_to_model_name.items()]
            # add the target model too; it's a smoke test that it should get a score of 0
            check_models.append((len(model_idx_to_model_name), target_name))

            for idx, name in check_models:
                baseline_score = None
                for row in cursor.execute('SELECT baseline_score FROM baseline_scores_by_source_model WHERE source_model = ? LIMIT 1', (name,)):
                    baseline_score = row[0]

                if baseline_score is None:
                    fut = executor.submit(evaluate_quant2_base_model, self, tensors_by_name, idx, name, max_workers=1)
                    futs.append(fut)

            for fut in tqdm.tqdm(cf.as_completed(futs), total=len(futs)):
                name, rmse = fut.result()
                cursor.execute('INSERT INTO baseline_scores_by_source_model (source_model, baseline_score) VALUES (?, ?)', (name, rmse))
                conn.commit()

        balanced_baseline_score = None

        for row in cursor.execute('SELECT balanced_baseline_score FROM balanced_baseline_score LIMIT 1'):
            balanced_baseline_score = row[0]
            print('Re-using balanced baseline score:', balanced_baseline_score)
            break

        if balanced_baseline_score is None:
            _, balanced_baseline_score = evaluate_quant2(self, None, source_type_to_index, tensors_by_name, src_vec, 0, max_workers=16)

            cursor.execute('INSERT INTO balanced_baseline_score (balanced_baseline_score) VALUES (?)', (balanced_baseline_score,))
            conn.commit()

        # Here is the scheme:
        # For every model we are using as a source, learn a multiplier.
        # Also learn a bias term.
        #
        # Dequant:   bias + w1 * tensor1 + w2 * tensor2 + ... + wn * tensorn
        #
        # Where the tensor1, tensor2 etc are dequantized tensors.
        #
        # And that's it.
        #
        # Optimized with CMA-ES ... because that felt easiest to code.
        # We are using the entire model for MSE calculations, and the cma
        # ask/tell interface, and some threads to make this use all the CPUs.

        es = cma.CMAEvolutionStrategy(src_vec, 0.02, {'popsize': 16})

        n_tensors = len(tensors_by_name)

        epoch = 0

        while True:
            epoch += 1
            tensor_idxs_for_this_round = np.random.choice(n_tensors, size=10, replace=False)

            candidates = es.ask()

            with cf.ThreadPoolExecutor() as executor:
                futures = [executor.submit(evaluate_quant2, self, tensor_idxs_for_this_round, source_type_to_index, tensors_by_name, candidate, idx) for idx, candidate in enumerate(candidates)]
                results = []
                for result in tqdm.tqdm(cf.as_completed(futures), total=len(futures)):
                    try:
                        results.append(result.result())
                    except Exception as e:
                        sys.stderr.write(f'Error: {e}\n')
                        raise

            results = sorted(results, key=lambda x: x[0])
            scores = [x[1] for x in results]
            print('Current scores:', scores)

            es.tell(candidates, scores)

            results = sorted(results, key=lambda x: x[1])
            print('Best result:', results[0])
            print('Best candidate:', candidates[results[0][0]])
            print('')
            print('')

            best_candidate = candidates[results[0][0]]

            model_json = {
                'bias': best_candidate[-1],
                'weights': dict([(k.name, best_candidate[v]) for k, v in source_type_to_index.items()])
            }

            rmse = results[0][1].item()
            rmse_random_baseline_adjusted = rmse / random_baseline_score
            rmse_balanced_baseline_adjusted = rmse / balanced_baseline_score

            cmaes_sigma = es.sigma

            cursor.execute('INSERT INTO train_results (model, rmse, random_baseline_adjusted_rmse_score, balanced_baseline_adjusted_rmse_score, epoch, cmaes_sigma) VALUES (?, ?, ?, ?, ?, ?)', (json.dumps(model_json, indent=4, sort_keys=True), rmse, rmse_random_baseline_adjusted, rmse_balanced_baseline_adjusted, epoch, cmaes_sigma))
            conn.commit()


    def apply(self, trained, tensors_by_name):
        raise UnimplementedError("Not implemented")

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

    if not models:
        print("Error: No models to train with.")
        sys.exit(1)

    tensors = set()
    target_reader = GGUFReader(target_model, 'r')
    for tensor in target_reader.tensors:
        tensors.add(tensor.name)
    model_readers = [GGUFReader(model, 'r') for model in models]

    model_idx_to_model_name = dict([(idx, model) for idx, model in enumerate(models)])

    tensors_by_name = {}
    for tensor in tensors:
        target_tensor = get_tensor(target_reader, tensor)
        model_tensors = []
        for model_reader in model_readers:
            model_tensor = get_tensor(model_reader, tensor)
            model_tensors.append(model_tensor)
        tensors_by_name[tensor] = (target_tensor, model_tensors)

    result = scheme.train(tensors_by_name, target_model, model_idx_to_model_name)

    print("Training complete: ", result)

if __name__ == '__main__':
    main()
