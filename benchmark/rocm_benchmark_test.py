from rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    from transformers import GPT2Tokenizer, FlaxGPTNeoModel
    from python_smi_tools.rocm_smi import getMaxPower
    print(getMaxPower(0))
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
    args = ROCmBenchmarkArguments(model=model, batch_sizes=[8], sequence_lengths=[8, 32, 128, 512],
                                  only_pretrain_model=True)
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)