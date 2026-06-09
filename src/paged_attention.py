import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from attention import MultiHeadAttention
from attention import generate
from attention import make_causal_mask

class BlockTable:

    def __init__(self, num_blocks, num_heads, block_size, head_dim, dtype=torch.float32, device="cpu"):

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
        self.cache_k = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)
        self.cache_v = torch.zeros((num_blocks, num_heads, block_size, head_dim), dtype=dtype, device=device)

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

@torch.no_grad()
def paged_generate(model, num_layers, input_seq, num_steps):
    # num_layers is unused here because we store raw embeddings (not per-layer projected K/V)
    # and our model is single-layer. In a real multi-layer transformer, you would instantiate
    # one PagedKVCache per layer and index into it by layer_idx during the forward pass.

    NUM_BLOCKS = 64
    BLOCK_SIZE = 128

    # Prefill Phase

    start_time = time.perf_counter()

    batch_size, seq_len, d_model = input_seq.shape
    device = input_seq.device

    # Our model's forward function, calculates attention and returns attention
    # scores output. It does not store the K, V value unless the cache is passed.
    # This technically wont work since PagedKVCache has different update function.
    # So we need to call the model without cache and extract the KV values
    # For this, we need to reconfigure blockTable to store embeddings not KV Heads
    # We do this by using num_heads = batch_size and head_dim = d_model
    # Each slot will then store a token embedding of shape (1, d_model)
    blockTable = BlockTable(num_blocks=NUM_BLOCKS, num_heads=batch_size, block_size=BLOCK_SIZE, head_dim=d_model, device=device)

    cache = PagedKVCache(blockTable, BLOCK_SIZE)
    seq_id = 0

    for i in range(seq_len):
        cache.update(seq_id, i, input_seq[:, i, :], input_seq[:, i, :])

    mask = make_causal_mask(seq_len, input_seq.device)
    output = model(input_seq, input_seq, input_seq, mask=mask)

    input_seq = torch.cat((input_seq, output[:, -1:, :]), dim=1)

    end_time = time.perf_counter()

    prefill_time = ((end_time-start_time)*1000)

    # Decode Phase

    decode_times = []

    for j in range(num_steps-1):
        start_time = time.perf_counter()

        cache.update(seq_id=seq_id, position=seq_len+j, key=input_seq[:, -1, :], value=input_seq[:, -1, :])
        
        k_all, v_all = cache.read_all(seq_id)
        # Since Cache will return BLOCK_SIZE slots, it will have zero-padded ones as well
        k_all = k_all[:, :seq_len+j+1, :]
        v_all = v_all[:, :seq_len+j+1, :]

        output = model(input_seq[:, -1:, :], k_all, v_all, mask=None)
        input_seq = torch.cat((input_seq, output[:, -1:, :]), dim=1)
        end_time = time.perf_counter()
        decode_times.append((end_time-start_time)*1000)

    return input_seq, prefill_time, decode_times

if __name__ == "__main__":

    NUM_LAYERS = 1
    BATCH_SIZE = 4
    NUM_HEADS = 8
    HEAD_DIM = 256

    print("\n####################################")
    print("#  KV Cache Generation Validation  #")
    print("####################################\n")

    BATCH_SIZE = 1
    SEQ_LEN = 10
    D_MODEL = 256
    NUM_LAYERS = 1
    STEPS = 50

    # Create a random input_seq
    input_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

    # Instantiate Model
    model = MultiHeadAttention(d_model=D_MODEL, num_heads=4)

    # Run Generate Function
    _, prefill_time, decode_times = paged_generate(model=model, num_layers=NUM_LAYERS, input_seq=input_seq, num_steps=STEPS)

    print(f"Total Prefill Time: {prefill_time} ms")
    print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
    print(f"Max Step Time: {max(decode_times)} ms")
    print(f"Min Step Time: {min(decode_times)} ms")
    print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

    print("\n\n#######################################")
    print("#  Generation allclose() Validation   #")
    print("#######################################\n")

    model = MultiHeadAttention(d_model=D_MODEL, num_heads=4)
    # Create a random input_seq
    input_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

    output_no_cache, times_no_cache = generate(model=model, input_seq=input_seq.clone(), num_steps=STEPS)

    output_cached, _, _ = paged_generate(model=model, num_layers=NUM_LAYERS, input_seq=input_seq.clone(), num_steps=STEPS)

    if torch.allclose(output_cached, output_no_cache, atol=1e-5):
        print("PASS: Outputs Match\n")
    else:
        print("FAIL: Outputs differ\n")