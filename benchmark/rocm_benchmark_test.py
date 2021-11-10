from rocm_benchmark import ROCmBenchmark, ROCmBenchmarkArguments

if __name__ == '__main__':
    from transformers import GPT2Tokenizer, FlaxGPTNeoModel

    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    model = FlaxGPTNeoModel.from_pretrained('EleutherAI/gpt-neo-1.3B')

    inputs = tokenizer("Hello, my dog is cute", return_tensors='jax')
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print("Outputs: {}".format(outputs))
    print("Outputs dict: {}".format(outputs.__dict__))
    print("Last hidden states: {}".format(last_hidden_states))
    args = ROCmBenchmarkArguments(models=['EleutherAI/gpt-neo-1.3B'], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512],
                                  only_pretrain_model=True)
    benchmark = ROCmBenchmark(args)
    results = benchmark.run()
    print(results)