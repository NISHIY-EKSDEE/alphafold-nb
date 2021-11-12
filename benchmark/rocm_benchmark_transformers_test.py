from benchmark_transformers.rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    from transformers import GPT2Tokenizer, FlaxGPTNeoModel

    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')

    args = ROCmBenchmarkArguments(model=model, models=['EleutherAI/gpt-neo-1.3B'], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512],
                                  multi_process=False)
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)