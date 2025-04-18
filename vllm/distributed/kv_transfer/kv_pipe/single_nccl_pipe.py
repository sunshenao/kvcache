# SPDX-License-Identifier: Apache-2.0
"""
    This module implements a PyNccl pipe for sending and receiving
    Optional[torch.Tensor] between distributed ranks with advanced
    communication features.

    Key Features:
    - Supports sending and receiving tensors with metadata
    - Handles both CUDA and CPU device communications
    - Implements a non-blocking tensor transfer mechanism
    - Manages buffer size and provides backpressure control
    - Supports distributed process groups with configurable parameters
"""

import datetime
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

import comm
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from torch.distributed import TCPStore,ReduceOp
from vllm.config import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (ncclRedOpTypeEnum,
    NCCLLibrary, buffer_type, cudaStream_t, ncclComm_t, ncclDataTypeEnum)
from vllm.utils import current_stream, get_ip

logger = init_logger(__name__)


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


Metadata = Dict[str, Optional[torch.Tensor]]


class SingleNcclPipe(KVPipeBase):

    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(self,
                 local_rank: int,
                 config: KVTransferConfig,
                 device: Optional[str] = None,
                 kv_match=None,
                 library_path: Optional[str] = None,
                 store = None):
        self.kv_match = kv_match
        self.config = config
        self.local_rank = local_rank
        self.kv_rank = self.config.kv_rank
        self.kv_parallel_size = self.config.kv_parallel_size

        self.nccl = NCCLLibrary(library_path)

        if device is None:
            self.device = self._select_device(self.config.kv_buffer_device)
        else:
            self.device = self._select_device(device)

        # build distributed connection and send/recv implementation
        
        self.store = store

        # self.rank = 1 if self.config.is_kv_consumer else 0
        self.group = StatelessProcessGroup.create_store(
            rank=self.kv_rank,
            world_size=self.kv_parallel_size,
            store=self.store,
        )
        # self.group.barrier()
        self.target_rank = self.kv_match[0] if self.kv_match[1] == self.kv_rank else self.kv_match[1]
        # self.comm = self.create_connect()

        impl = self._get_device_send_recv_impl(self.group)
        self.device_send_func, self.device_recv_func = impl
        
        # transportation-related variables
        self.transport_thread_recv: Optional[ThreadPoolExecutor] = None
        self.transport_thread_send: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()
        self.buffer_size_thresh = self.config.kv_buffer_size
        self.cuda_stream = torch.cuda.Stream()


    def _get_device_send_recv_impl(
        self, group: StatelessProcessGroup
    ) -> Tuple[Callable[[torch.Tensor, int], None], Callable[
        [torch.Tensor, int], None]]:
        send: Callable[[torch.Tensor, int], None]
        recv: Callable[[torch.Tensor, int], None]
        if self.device.type == "cuda":
            self.create_connect()
            self.group.clear()

            send, recv = self._send, self._recv  # type: ignore
        else:
            send = group.send_obj
            def my_recv(x, src):
                x[...] = group.recv_obj(src)
            recv = my_recv
        return send, recv

    def _select_device(self, device: str):
        logger.info("Selecting device: %s", device)
        if device == "cuda":
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def create_connect(self):
      
        if self.config.kv_rank == self.kv_match[0]:
            rank = 0
            self.unique_id = self.nccl.ncclGetUniqueId()
            self._send_metadata(self.unique_id)
        else:
            rank = 1
            self.unique_id = self._recv_metadata(True)
            # self.
        with torch.cuda.device(self.device):
            #print(self.unique_id,self.kv_rank)
            self.comm = self.nccl.ncclCommInitRank(2,self.unique_id,rank)

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=self.device)
            self.all_reduce(data)
            stream.synchronize()

            del data

            logger.info(f"{self.kv_match[0]}_{self.kv_match[1]}连接成功")

    def _send_metadata(self, metadata):
        # print("_send_metadata",self.target_rank)
        # target_rank = self.kv_match[0] if self.config.is_kv_consumer else self.kv_match[1]
        self.group.send_obj(metadata, self.target_rank)

    def _recv_metadata(self,delete=False):
        # target_rank = self.kv_match[0] if self.config.is_kv_consumer else self.kv_match[1]
        return self.group.recv_obj(self.target_rank,delete)

    

    def _send(self, tensor: torch.Tensor,  stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream =  self.cuda_stream

        dst = 1 if self.config.is_kv_producer else 0

        self.nccl.ncclSend(buffer_type(tensor.data_ptr()), tensor.numel(),
                            ncclDataTypeEnum.from_torch(tensor.dtype), dst,
                            self.comm, cudaStream_t(stream.cuda_stream))

    def _recv(self, tensor: torch.Tensor, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}")
        if stream is None:
            stream =  current_stream()   #self.cuda_stream # current_stream()

        src = 1 if self.config.is_kv_producer else 0

        self.nccl.ncclRecv(buffer_type(tensor.data_ptr()), tensor.numel(),
                            ncclDataTypeEnum.from_torch(tensor.dtype), src,
                            self.comm, cudaStream_t(stream.cuda_stream))

    
    def _make_metadata(self, tensor: Optional[torch.Tensor]) -> Metadata:
        
        if tensor is None:
            return {"dtype": None, "shape": None}
        else:
            return {"dtype": tensor.dtype, "shape": tensor.shape}
        

    def _prepare_recv_buffer(self, metadata: Metadata) -> torch.Tensor:
        
        return torch.empty(metadata["shape"],
                           dtype=metadata["dtype"],
                           device=self.device)
    

    def _send_impl(self,tensor: Optional[torch.Tensor]):
        if tensor is None:
            return 
        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        self._send(tensor.to(self.device)) # self.target_rank
    
    def _recv_impl(self) -> Optional[torch.Tensor]:
        metadata = self._recv_metadata()
        if metadata["dtype"] is None:
            return None
        buffer = self._prepare_recv_buffer(metadata)
        
        self._recv(buffer)

        return buffer
    
    def send_tensor_wrapper(self, tensor: Optional[torch.Tensor],
                            tensor_size: int) -> None:

        try:
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size -= tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):

        while self.buffer_size > self.buffer_size_thresh:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.01)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:

        if self.transport_thread_send is None:
            self.transport_thread_send = ThreadPoolExecutor(max_workers=1)

        if tensor is not None:
            tensor_size = tensor.element_size() * tensor.numel()
        else:
            tensor_size = 0

        self.block_if_full()
        with self.buffer_size_lock:
            self.buffer_size += tensor_size
        # self.send_tensor_wrapper(tensor,tensor_size)
        #print("send_tensor")
        self.transport_thread_send.submit(self.send_tensor_wrapper, tensor,
                                     tensor_size)


    def recv_tensor(self) -> Optional[torch.Tensor]:
        if self.transport_thread_recv is None:
            self.transport_thread_recv = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread_recv.submit(self._recv_impl)

        try:
            tensor = future.result()  # self._recv_impl()# future.result()
        except Exception as e:
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            logger.error("My device: %s", self.device)
            import traceback
            traceback.print_exc()
            raise e
        # 
        return tensor

    def all_reduce(self,
                   in_tensor: torch.Tensor,
                   op: ReduceOp = ReduceOp.SUM,
                   stream=None) -> torch.Tensor:
        
        # nccl communicator created on a specific device
        # will only work on tensors on the same device
        # otherwise it will cause "illegal memory access"
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}")

        out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()
        self.nccl.ncclAllReduce(buffer_type(in_tensor.data_ptr()),
                                buffer_type(out_tensor.data_ptr()),
                                in_tensor.numel(),
                                ncclDataTypeEnum.from_torch(in_tensor.dtype),
                                ncclRedOpTypeEnum.from_torch(op), self.comm,
                                cudaStream_t(stream.cuda_stream))
        return out_tensor
    

    def close(self):

        if hasattr(self,
                   "transport_thread_send") and self.transport_thread_send is not None:
            self.transport_thread_send.shutdown()

        if hasattr(self,
                   "transport_thread_recv") and self.transport_thread_recv is not None:
            self.transport_thread_send.shutdown()

        self.nccl.ncclCommDestroy(self.comm); 
    