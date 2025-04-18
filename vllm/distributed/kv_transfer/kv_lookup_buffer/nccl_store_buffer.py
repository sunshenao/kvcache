# SPDX-License-Identifier: Apache-2.0
"""
    Implements a distributed key-value (KV) cache transfer mechanism.

    Key Features:
    - Distributed KV cache transmission using PyNccl pipes.
    - Non-blocking `insert`, blocking `drop_select`.
    - Use CPU signal pipe to avoid racing condition
    - Handles buffer size constraints and provide backpressure mechanism to
      stop the prefill instance when the decode instance is slow.
"""
from ast import Dict
import threading
from collections import defaultdict, deque
from typing import Deque, List, Optional, Union

from matplotlib import axis
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase)
from vllm.distributed.kv_transfer.kv_pipe.base import KVPipeBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class NcclStoreBuffer(KVLookupBufferBase):

    def __init__(self, signal_pipes: dict[int,KVPipeBase], data_pipes: dict[int,KVPipeBase],
                 buffer_size_thresh: float,config: KVTransferConfig ):
        """
        signal_pipe: on CPU

        NOTE: on-device recv will block all threads in the process, making the
        KV cache producer unable to listen to new request while transmitting
        KV cache. Luckily CPU recv only blocks the current thread so we use
        CPU recv to listen to new request.

        data_pipe: on device (e.g. GPU)
        """
        self.kv_config = config
        # self.buffer: Deque[List[torch.Tensor]] = deque()
        self.buffer: Dict[str,List[torch.Tensor]] = defaultdict(lambda:None)

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh*0.9
        self.buffer_cv = threading.Condition()
        self.signal_pipes = signal_pipes
        self.data_pipes = data_pipes
        self.request_handling_thread: dict[str,Optional[threading.Thread]] = defaultdict(lambda:None)

        self.normal_signal = torch.tensor([0], device="cpu")
        self.end_signal = None

        self.t = 0
        self.t1 = 0

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor],target:int) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        if tensor.dtype == torch.bool:
            tensor = tensor.float()
        self.data_pipes[target].send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError(f"Unknown data type {type(data)}")

    def _add_to_buffer(self,key_world:str , key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [key, value, hidden]
        data_size = sum([self._get_element_size(data) for data in buffer_item])

        with self.buffer_cv:
            if self.buffer_size + data_size > self.buffer_size_threshold:
                # log outside the while loop to avoid this message being logged
                # repeatedly.
                logger.debug("KV transfer buffer is full. Handling...")
                while self.buffer_size + data_size > self.buffer_size_threshold:
                    self.buffer_cv.wait()

            self.buffer_size += data_size
            self.buffer[key_world] = buffer_item
            self.buffer_cv.notify_all()

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self,target_rank):

        try:
            while True:
                if self.kv_config.is_kv_consumer:
                    self.request_handling_thread[target_rank] = None
                    return 
                key_world = self.signal_pipes[target_rank]._recv_metadata()
                def is_buffer_available(
                    key_world: str ) -> bool:
                    # perform input tokens and roi matching
                    # FIXME: this matching is O(n), ideally it should be O(1)
                    # but this buffer size won't (and shouldn't) be too large so
                    # the fix is not urgent.
                    if key_world in self.buffer.keys():
                        return True
                    return False
         
                with self.buffer_cv:
                    while not is_buffer_available(key_world):
                        logger.debug(
                            "KV transfer buffer is not available. Waiting...")
                        self.buffer_cv.wait()
                    # need to clone the tensor
                    # in case the tensor is freed before sending finishes
                    matched_item = self.buffer.pop(key_world)

                    # layer,seq_len,head,dim = matched_item[0].shape
                    # hidden = matched_item[-1].reshape(seq_len,layer,1,dim).permute(1,0,2,3)
                    # transfer_data = torch.cat([matched_item[0],matched_item[1],hidden],axis=-2)
                    
                    for tensor in matched_item:
                        self._send_tensor_and_dec_size(tensor=tensor,target=target_rank)
                    # self._send_tensor_and_dec_size(tensor=transfer_data,target=target_rank)
                    self.buffer_cv.notify_all()

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select_handler_step(self,target_rank):
        key_world = self.signal_pipes[target_rank]._recv_metadata()
        def is_buffer_available(
            key_world: str ) -> bool:
            # perform input tokens and roi matching
            # FIXME: this matching is O(n), ideally it should be O(1)
            # but this buffer size won't (and shouldn't) be too large so
            # the fix is not urgent.
            if key_world in self.buffer.keys():
                return True
            return False

        with self.buffer_cv:
            while not is_buffer_available(key_world):
                logger.debug(
                    "KV transfer buffer is not available. Waiting...")
                self.buffer_cv.wait()
            # need to clone the tensor
            # in case the tensor is freed before sending finishes
            matched_item = self.buffer.pop(key_world)
            for tensor in matched_item:
                self.data_pipes[target_rank].send_tensor(tensor=tensor)
            self.buffer_cv.notify_all()

    def drop_select(
            self, key_world:str,target_rank:int) -> List[Optional[torch.Tensor]]:
        
        # key_pipe = str(kv_match[0])+'_'+str(kv_match[1])
        assert self.request_handling_thread[target_rank] is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"
        self.signal_pipes[target_rank]._send_metadata(key_world)
        # self.data_pipes[key_pipe].send_tensor(input_tokens)
        # self.data_pipes[key_pipe].send_tensor(roi)

        # input_tokens = self.data_pipes[key_pipe].recv_tensor()
        # roi = self.data_pipes[key_pipe].recv_tensor()
        # if roi is not None:
        #     # convert from float tensor to bool tensor
        #     # as PyNccl does not support sending bool tensor
        #     roi = (roi > 0.5)
        # data = self.data_pipes[target_rank].recv_tensor()
        # layer,seq_len,head,dim = data.shape
        # hidden = data[:,:,-1,:].permute(1,0,2).reshape((seq_len,-1))
        # key = data[:,:,:head//2,:]
        # value = data[:,:,head//2:head-1,:]
        key = self.data_pipes[target_rank].recv_tensor()
        value = self.data_pipes[target_rank].recv_tensor()
        hidden = self.data_pipes[target_rank].recv_tensor()


        return [key, value, hidden]

    def insert(self,key_world:str, 
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor,target_rank:int) -> None:
        
        self._add_to_buffer(key_world, key, value, hidden)

        if self.request_handling_thread[target_rank] is None:
            self.request_handling_thread[target_rank] = threading.Thread(
                target=self.drop_select_handler,args=(target_rank,))
            self.request_handling_thread[target_rank].start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            for _,thread in self.request_handling_thread:
                thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
