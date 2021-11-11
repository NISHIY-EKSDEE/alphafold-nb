from __future__ import absolute_import
import sys
sys.path.append("../..")
sys.path.append("..")
import timeit
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from transformers.benchmark.benchmark_args_utils import BenchmarkArguments
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import is_py3nvml_available, is_torch_available, cached_property
from transformers.models.auto.modeling_auto import MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from transformers.utils import logging
from transformers.benchmark.benchmark_utils import (
    Benchmark,
    Memory,
    MemorySummary,
    measure_peak_memory_cpu,
    start_memory_tracing,
    stop_memory_tracing,
)

from jax import random
import jax
from multiprocessing import Pipe, Process

from benchmark.python_smi_tools.rocm_smi import getMemInfo, initializeRsmi

key = random.PRNGKey(0)
logger = logging.get_logger(__name__)


@dataclass
class ROCmBenchmarkArguments(BenchmarkArguments):

    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or {positive_arg}={kwargs[positive_arg]}"
                )
        self.model = kwargs.pop("model", None)
        self.do_multi_processing = kwargs.pop("do_multi_processing", None)
        self.torchscript = kwargs.pop("torchscript", self.torchscript)
        self.torch_xla_tpu_print_metrics = kwargs.pop("torch_xla_tpu_print_metrics", self.torch_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        if kwargs.get('models') is None and self.model is not None:
            kwargs['models'] = ['bert-base-uncased']

        super().__init__(**kwargs)

    torchscript: bool = field(default=False, metadata={"help": "Trace the models using torchscript"})
    torch_xla_tpu_print_metrics: bool = field(default=False, metadata={"help": "Print Xla/PyTorch tpu metrics"})
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    @cached_property
    def _setup_devices(self) -> Tuple["jax.device", int]:
        return jax.devices()[0].id, len(jax.devices())

    @property
    def is_tpu(self):
        return False and self.tpu

    @property
    def device_idx(self) -> int:
        # TODO(PVP): currently only single GPU is supported
        return jax.devices()[0]

    @property
    def device(self) -> "jax.device":
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        return self.n_gpu > 0


class ROCmBenchmark(Benchmark):

    args: ROCmBenchmarkArguments
    configs: PretrainedConfig
    framework: str = "PyTorch"

    @property
    def framework_version(self):
        return None

    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_inference)

    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        _inference = self._prepare_inference_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_inference)

    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_speed(_train)

    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        _train = self._prepare_train_func(model_name, batch_size, sequence_length)
        return self._measure_memory(_train)

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]
        if self.args.model:
            model = self.args.model
        else:
            if self.args.torchscript:
                config.torchscript = True

            has_model_class_in_config = (
                hasattr(config, "architectures")
                and isinstance(config.architectures, list)
                and len(config.architectures) > 0
            )
            if not self.args.only_pretrain_model and has_model_class_in_config:
                try:
                    model_class = config.architectures[0]
                    transformers_module = __import__("transformers", fromlist=[model_class])
                    model_cls = getattr(transformers_module, model_class)
                    model = model_cls(config)
                except ImportError:
                    raise ImportError(
                        f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                    )
            else:
                model = MODEL_MAPPING[config.__class__](config)
            model.eval()
            model.to(self.args.device)
        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size
        input_ids = np.random.randint(low=0, high=11111, size=(batch_size, sequence_length))

        inference_model = model

        def encoder_decoder_forward():
            outputs = inference_model(input_ids, decoder_input_ids=input_ids)
            return outputs

        def encoder_forward():
            outputs = inference_model(input_ids)
            return outputs
        _forward = encoder_decoder_forward if config.is_encoder_decoder else encoder_forward
        return _forward

    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        config = self.config_dict[model_name]

        has_model_class_in_config = (
            hasattr(config, "architectures")
            and isinstance(config.architectures, list)
            and len(config.architectures) > 0
        )
        if not self.args.only_pretrain_model and has_model_class_in_config:
            try:
                model_class = config.architectures[0]
                transformers_module = __import__("transformers", fromlist=[model_class])
                model_cls = getattr(transformers_module, model_class)
                model = model_cls(config)
            except ImportError:
                raise ImportError(
                    f"{model_class} does not exist. If you just want to test the pretrained model, you might want to set `--only_pretrain_model` or `args.only_pretrain_model=True`."
                )
        else:
            model = MODEL_WITH_LM_HEAD_MAPPING[config.__class__](config)

        if self.args.torchscript:
            raise NotImplementedError("Training for torchscript is currently not implemented")
        else:
            train_model = model

        model.train()
        model.to(self.args.device)

        # encoder-decoder has vocab size saved differently
        vocab_size = config.vocab_size if hasattr(config, "vocab_size") else config.encoder.vocab_size

        input_ids = random.randint(key, (batch_size, sequence_length), dtype=jax.numpy.long, minval=0,
                                   maxval=vocab_size)

        if self.args.fp16:
            logger.info("Running training in Mixed Precision...")
            assert self.args.is_gpu, "Mixed precision is possible only for GPU."

            # amp seems to have memory leaks so that memory usage
            # is measured using .half() for now https://github.com/NVIDIA/apex/issues/439
            model.half()

        def compute_loss_and_backprob_encoder():
            loss = train_model(input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        def compute_loss_and_backprob_encoder_decoder():
            loss = train_model(input_ids, decoder_input_ids=input_ids, labels=input_ids)[0]
            loss.backward()
            return loss

        _train = (
            compute_loss_and_backprob_encoder_decoder
            if config.is_encoder_decoder
            else compute_loss_and_backprob_encoder
        )
        return _train

    def _measure_speed(self, func) -> float:
        try:
            if self.args.is_tpu or self.args.torchscript:
                # run additional 10 times to stabilize compilation for tpu and torchscript
                logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                timeit.repeat(
                    func,
                    repeat=1,
                    number=5,
                )

            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
            runtimes = timeit.repeat(
                func,
                repeat=self.args.repeat,
                number=10,
            )

            return min(runtimes) / 10.0
        except RuntimeError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A"

    def _measure_memory(self, func: Callable[[], None]) -> [Memory, MemorySummary]:

        try:
            if self.args.is_gpu:
                logger.info(
                    "Measuring total GPU usage on GPU device. Make sure to not have additional processes running on the same GPU."
                )

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
                    trace_rocm_memory_porcess = Process(target=trace_rocm_memory, args=(self.args.device, receiver))
                    trace_rocm_memory_porcess.start()

                    func()

                    sender.send(0)
                    memory_trace = sender.recv()
                    trace_rocm_memory_porcess.join()
                except:
                    trace_rocm_memory_porcess.terminate()

                # max_bytes_in_use = max(memory_trace)
                memory = Memory(0)
            else:
                # cpu
                memory_bytes = measure_peak_memory_cpu(func)
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int) else memory_bytes
            return memory, None
        except RuntimeError as e:
            self.print_fn(f"Doesn't fit on GPU. {e}")
            return "N/A", None



