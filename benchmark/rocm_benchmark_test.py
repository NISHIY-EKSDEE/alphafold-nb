from .rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    args = ROCmBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)