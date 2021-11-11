from rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    from transformers import GPT2Tokenizer, FlaxGPTNeoModel
    from python_smi_tools.rocm_smi import getMaxPower, showMemInfo, initializeRsmi
    import jax
    import numpy as np

    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')



    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
    inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
    print(inputs)
    res = model(np.random.randint(low=0, high=11111, size=(2, 25)))
    print(res)
    print('fail')
    exit()

    args = ROCmBenchmarkArguments(model=model, batch_sizes=[2], sequence_lengths=[8, 32, 128, 512],)
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)