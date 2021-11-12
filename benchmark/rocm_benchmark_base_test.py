from benchmark_base.rocm_benchmark import run_rocm_benchmark

if __name__ == '__main__':
    from transformers import GPT2Tokenizer, FlaxGPTNeoModel

    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
    inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
    run_rocm_benchmark(model, inputs)