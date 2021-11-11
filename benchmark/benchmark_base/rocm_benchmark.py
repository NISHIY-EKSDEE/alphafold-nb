import timeit
from multiprocessing import Pipe, Process
import jax
from benchmark.python_smi_tools.rocm_smi import getMemInfo, initializeRsmi


def print_result(memory, compute_time):
    print("\n" + 20 * "=" + ("INFERENCE - MEMOMRY - SPEED - SUMMARY").center(40) + 20 * "=")
    print("Time in s".center(15) + "Memory in MB".center(15))
    print(str(compute_time).center(15) + str(memory).center(15))


def run_rocm_benchmark(model, input):
    memory, compute_time = None, None
    def forward_func():
        return model(**input)

    runtimes = timeit.repeat(
        forward_func,
        repeat=5,
        number=10,
    )
    compute_time = min(runtimes) / 10.0

    def trace_rocm_memory(device, exit_pipe, interval=0.5):
        memory_trace = []
        while True:
            memory_trace.append(getMemInfo(device, 'vram')[0])
            if exit_pipe.poll(interval):
                exit_pipe.send(memory_trace)
                return 0
    initializeRsmi()
    receiver, sender = Pipe()
    try:
        trace_rocm_memory_porcess = Process(target=trace_rocm_memory, args=(jax.devices()[0].id, receiver))
        trace_rocm_memory_porcess.start()

        forward_func()
        sender.send(0)
        memory_trace = sender.recv()
        trace_rocm_memory_porcess.join()
        max_bytes_in_use = max(memory_trace)
        memory = max_bytes_in_use / (1024**2)
    except:
        trace_rocm_memory_porcess.terminate()
    print_result(memory, compute_time)
    return memory, compute_time