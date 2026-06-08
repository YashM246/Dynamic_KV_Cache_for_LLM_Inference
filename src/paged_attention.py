import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BlockTable:

    def __init__(self, num_blocks, num_heads, block_size, head_dim, dtype=torch.float32):

        # Args:
        #       num_blocks -> Number of blocks in the pool (Determines total mem capacity)
        #       num_heads -> Number of KV Heads in the Model
        #       block_size -> Number of tokens that fit in one block
        #       head_dim -> Dimension size of each attention head

        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype

        # Initialize K Values Tensor and a V Values Tensor
        self.cache_k = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)
        self.cache_v = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype)

        # List to Track Free Blocks
        self.free_list = torch.arange(num_blocks)

    def allocate(self,):
        # Check if not empty
        if self.free_list.numel() == 0:
            raise ValueError("Out of Memory T_T")

        indx = self.free_list[0].item()
        self.free_list = self.free_list[1:]
        return indx
        
    def free(self, block_idx):
        self.free_list = torch.cat((self.free_list, torch.tensor([block_idx])), dim=0)

    def write(self, block_idx, slot, key, value):
        self.cache_k[block_idx, :, slot, :] = key
        self.cache_v[block_idx, :, slot, :] = value

    def read(self, block_idx):
        return self.cache_k[block_idx], self.cache_v[block_idx]
    

class PagedKVCache:

    def __init__(self, block_table, block_size):

        self.block_table = block_table
        self.block_size = block_size

        self.seq_map = {}
        # Sequence Map -> seq_id: list of allocated block indices

    def update(self, seq_id, position, key, value):

        # Need to find which block in the sequence this position belongs to
        block_num = position // self.block_size
        slot = position % self.block_size

        # If sequence does not have enough blocks, allocate one
        if seq_id not in self.seq_map:  
            self.seq_map[seq_id] = []

        if len(self.seq_map[seq_id]) <= block_num:
            new_block = self.block_table.allocate()
            self.seq_map[seq_id].append(new_block)

        # Write to the correct physical block
        physical_block = self.seq_map[seq_id][block_num]

        self.block_table.write(physical_block, slot, key, value)

    def read_all(self, seq_id):
        
        # Collect all blocks in the sequence and concat

        list_k, list_v = [], []
        for block_idx in self.seq_map[seq_id]:
            K, V = self.block_table.read(block_idx)
            list_k.append(K)
            list_v.append(V)
        tensor_k = torch.cat(list_k, dim=1)
        tensor_v = torch.cat(list_v, dim=1)

        return tensor_k, tensor_v
    
    def free_sequence(self, seq_id):

        for block_idx in self.seq_map[seq_id]:
            self.block_table.free(block_idx)
        
        del self.seq_map[seq_id]
