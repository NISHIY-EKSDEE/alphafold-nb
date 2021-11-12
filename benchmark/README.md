# Benchmark transformers for ROCm
Used default `transformers` benchmark architecture, but 
Required [ROCm SMI lib](https://github.com/RadeonOpenCompute/rocm_smi_lib)
### Usage
Working only in **single process**! Use `multi_process=False`
```python
from benchmark_transformers.rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments
from transformers import GPT2Tokenizer, FlaxGPTNeoModel

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')

args = ROCmBenchmarkArguments(model=model, batch_sizes=[2], sequence_lengths=[8, 32],
                              multi_process=False)
benchmark = ROCmBenchmark(args)
results = benchmark.run()

```
Test example in `benchmark/rocm_benchmark_transformers_test.py`

# Benchmark base for ROCm
A simple benchmark for different models
### Usage
```python
from benchmark_base.rocm_benchmark import run_rocm_benchmark
from transformers import GPT2Tokenizer, FlaxGPTNeoModel

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
result = run_rocm_benchmark(model, inputs)
```
Test example in `benchmark/rocm_benchmark_base_test.py`


