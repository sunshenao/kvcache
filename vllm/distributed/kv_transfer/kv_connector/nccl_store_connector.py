# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from csv import Error
import datetime
import hashlib
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import redis
import torch

from tools.report_build_time_ninja import Target
from vllm import _custom_ops as ops, envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.nccl_store_buffer import (
    NcclStoreBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from torch.distributed import TCPStore,ReduceOp
from vllm.distributed.kv_transfer.kv_pipe.single_nccl_pipe import (SingleNcclPipe)
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class NcclStoreConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):  
        self.local_rank = local_rank
        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size
        self.is_deepseek_mla = config.model_config.is_deepseek_mla
        self.use_mla_opt = not envs.VLLM_MLA_DISABLE


        self.lookup_buffer_size = self.config.kv_buffer_size

        self.buffer: Optional[NcclStoreBuffer] = None

        self.data_pipe: SingleNcclPipe
        self.signal_pipe: SingleNcclPipe

        # 2 pipes for every rank in the world
        port_offset_base = 2 * self.config.kv_parallel_size
        # self.kv_matchs = self.config.kv_matchs

        # 设置存储数据库
        
        # self.store = redis.Redis(host="localhost", port=6379)
        # if self.config.kv_rank == 0:
        #     self.store.flushall()
        # pubsub = self.store.pubsub()
        # my_channel = f"channel_{self.config.kv_rank}" # 监听频道
        # pubsub.subscribe(my_channel)


        self.data_pipes = {}
        self.signal_pipes = {}
        self.buffers = {}

        self.kv = self.mix_list(self.config.kv_parallel_size)

        store_timeout = self.config.get_from_extra_config("store_timeout", 60)
        self.store = []
        # for index,kv_match in enumerate(self.kv):
        #     if not self.config.kv_rank in kv_match:
        #         continue
            

        for index,kv_match in enumerate(self.kv):
            
            if not self.config.kv_rank in kv_match:
                continue
            target_rank = kv_match[0] if kv_match[1] == self.config.kv_rank else kv_match[1]
            
            store = TCPStore(
                host_name=self.config.kv_ip,
                port=self.config.kv_port + port_offset_base+(index+1+local_rank)*123,# 匹配的两个应该相同
                world_size=2,
                is_master=(self.config.kv_rank == kv_match[0]),
                timeout=datetime.timedelta(seconds=store_timeout),
            )
            self.store.append(store)
        
            self.data_pipe = SingleNcclPipe(
                local_rank=local_rank,
                config=self.config,
                store=store,
                kv_match=kv_match,
            )

            store1 = TCPStore(
                host_name=self.config.kv_ip,
                port=self.config.kv_port + port_offset_base+(index+1+local_rank)*123+1,# 匹配的两个应该相同
                world_size=2,
                is_master=(self.config.kv_rank == kv_match[0]),
                timeout=datetime.timedelta(seconds=store_timeout),
            )

            self.store.append(store1)
            self.signal_pipe = SingleNcclPipe(
                local_rank=local_rank,
                config=self.config,
                store=store1,
                device="cpu",
                kv_match=kv_match,
            )
            
            self.data_pipes[target_rank] = self.data_pipe
            self.signal_pipes[target_rank] = self.signal_pipe

        self.buffer = NcclStoreBuffer(self.signal_pipes,
                                        self.data_pipes,
                                        self.config.kv_buffer_size,
                                        self.config)
                
        

    def mix_list(self,world):
        res = []
        for i in range(world):
            for j in range(i+1,world):
                if i != j:
                    res.append([i,j])
        return res
    
    def select(self,key_world:str,target_rank:int) -> List[Optional[torch.Tensor]]:

        assert self.buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.buffer.drop_select(key_world, target_rank)

    def insert(self,key_world:str,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor,target_rank:int) -> None:

        assert self.buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.buffer.insert(key_world, key, value, hidden,target_rank=target_rank)

    def key(self,kv_match):
        return str(kv_match[0]) + '_' + str(kv_match[1])

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        self.to_producer()

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads


        request_ids = list(model_input.request_ids_to_seq_ids.keys())


        if self.is_deepseek_mla and self.use_mla_opt:
            head_size = model_config.kv_lora_rank + \
                model_config.qk_rope_head_dim
            num_heads = 1
        elif self.is_deepseek_mla and not self.use_mla_opt:
            head_size = model_config.qk_nope_head_dim + \
                model_config.qk_rope_head_dim
        else:
            head_size = getattr(model_config, "head_dim",
                                int(hidden_size // num_attention_heads))

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You have some decode requests while using "
                               "SimpleConnector. Their KVCache won't be sent.")
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                if self.is_deepseek_mla and self.use_mla_opt:
                    key_cache = kv_cache.reshape(-1, num_heads, head_size)
                    value_cache = kv_cache.reshape(-1, num_heads, head_size)
                else:
                    key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                    value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            key_world = self.tensor_hash(current_tokens[torch.ones_like(current_tokens,dtype=bool)],self.local_rank)

            target_rank = self.parse_request_id(request_id=request_ids[idx],is_prefill=True)
            self.insert(key_world, keys, values,
                        hidden_or_intermediate_states[start_pos:end_pos],
                        target_rank=target_rank)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        # self.to_consumer()
        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        model_config = model_executable.model.config

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        request_ids = list(model_input.request_ids_to_seq_ids.keys())
        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen
            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to --max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            target_rank = self.parse_request_id(request_id=request_ids[idx],is_prefill=False)

            key_world = self.tensor_hash(current_tokens[torch.ones_like(current_tokens,dtype=bool)],self.local_rank)
            ret = self.select(key_world,target_rank=target_rank)
            # print("select")
            if ret is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue
            keys: torch.Tensor = ret[0]
            values: torch.Tensor = ret[1]
            hidden: torch.Tensor = ret[2]

            num_computed_tokens = keys.shape[1]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                if self.is_deepseek_mla and self.use_mla_opt:
                    layer.self_attn.attn = layer.self_attn.mla_attn
                    k_c_normed_k_pe = keys[
                        i - model_executable.model.start_layer].to(
                            kv_cache.device).squeeze(1)
                    k_c_normed = k_c_normed_k_pe[:, :model_config.kv_lora_rank]
                    k_pe = k_c_normed_k_pe[:, model_config.kv_lora_rank:]
                    ops.concat_and_cache_mla(
                        k_c_normed,
                        k_pe,
                        kv_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale,
                    )
                else:
                    key_cache, value_cache = kv_cache[0], kv_cache[1]
                    ops.reshape_and_cache_flash(
                        keys[i - model_executable.model.start_layer].to(
                            key_cache.device),
                        values[i - model_executable.model.start_layer].to(
                            value_cache.device),
                        key_cache,
                        value_cache,
                        slot_mapping[start_pos:end_pos],
                        layer.self_attn.attn.kv_cache_dtype,
                        layer.self_attn.attn._k_scale,
                        layer.self_attn.attn._v_scale,
                    )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None
            print("失败： ",'--'*18)
            raise Error

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        
        for data_pipes in self.data_pipes:
            data_pipes.close()
        for signal_pipes in self.signal_pipes:
            signal_pipes.close()
        for store in self.store:
            store.finalize()
    
    def to_consumer(self):
        if self.config.is_kv_consumer:
            return 
        self.config.kv_role = "kv_consumer"
        
    def to_producer(self):
        if self.config.is_kv_producer:
            return 
        self.config.kv_role = "kv_producer"


    @staticmethod
    def tensor_hash(tensor: torch.Tensor,local_rank:int) -> int:
        """Calculate the hash value of the tensor."""
        tensor_bytes = tensor.clone().detach().cpu().numpy().tobytes()
        hash_object = hashlib.blake2b(tensor_bytes)
        hash_hex = hash_object.hexdigest()
        return int(hash_hex[:16], 16)+local_rank
    
    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True)->int:
        # logger.info("parse_request_id, request_id: %s, is_prefill: %s",
        #             request_id, is_prefill)
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(\d+)"
        else:
            pattern = r"___prefill_addr_(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            target_rank = match.group(1)
        
            return int(target_rank)
        raise ValueError(
            f"Request id {request_id} does not contain decode and prefill")
    