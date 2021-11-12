import argparse

from benchmark_transformers.rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    from transformers import FlaxPreTrainedModel
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()

    model = FlaxPreTrainedModel.from_pretrained(args.model_name)

    args = ROCmBenchmarkArguments(model=model, models=[args.model_name], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512],
                                  multi_process=False)
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)