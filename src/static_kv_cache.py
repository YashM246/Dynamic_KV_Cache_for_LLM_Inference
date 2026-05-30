import torch
import torch.nn as nn

class Static_KV_Cache():

    def __init__(self, num_layers, batch_size, num_heads, max_seq_len, head_dim, dtype=torch.float32):

        self.num_layers = num_layers        # Number of layers in the model
        self.batch_size = batch_size        # Batch size for Inference
        self.num_heads = num_heads          # Number of KV Heads
        self.max_seq_len = max_seq_len      # Maximum Sequence Length of Cache
        self.head_dim = head_dim            # Dimension of each Attention Head
        self.dtype = dtype                  # Data Type for Cache Tensors

        self.cache = {}                     # Actual KV Cache
        self.pos = 0                        # Position Pointer to track 

        for layer_idx in range(num_layers):
            self.cache[layer_idx] = {
                "key": torch.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype),
                "value": torch.zeros((batch_size, num_heads, max_seq_len, head_dim), dtype=dtype)
            }

    def update(self, layer_idx, new_key, new_value):
        
        current_keys = self.cache[layer_idx]["key"]
        current_values = self.cache[layer_idx]["value"]

        current_keys[:, :, self.pos, :] = new_key
        current_values[:, :, self.pos, :] = new_value
        self.pos += 1

        return current_keys[:, :, :self.pos, :], current_values[:, :, :self.pos, :]
    
    def reset(self,):
        self.pos = 0
        # New writes will override the data anyway

if __name__ == "__main__":

    NUM_LAYERS = 1
    BATCH_SIZE = 4
    NUM_HEADS = 8
    MAX_SEQ_LEN = 128
    HEAD_DIM = 256

    kv_cache = Static_KV_Cache(NUM_LAYERS, BATCH_SIZE, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)

    print(f"KV Cache Key Shape after Initialization: {kv_cache.cache[0]["key"].shape}")
    print(f"KV Cache Val Shape after Initialization: {kv_cache.cache[0]["value"].shape}")

    inp_k = torch.full((BATCH_SIZE, NUM_HEADS, HEAD_DIM), fill_value=6.0)
    inp_v = torch.full((BATCH_SIZE, NUM_HEADS, HEAD_DIM), fill_value=7.0)

    cached_k, cached_v = kv_cache.update(0, inp_k, inp_v)

    print(f"Returned Cache Key Shape after 1st Update: {cached_k.shape}")
    print(f"Returned Cache Val Shape after 1st Update: {cached_v.shape}")

    cached_k, cached_v = kv_cache.update(0, inp_k, inp_v)

    print(f"Returned Cache Key Shape after 2nd Update: {cached_k.shape}")
    print(f"Returned Cache Val Shape after 2nd Update: {cached_v.shape}")
    
    cached_k, cached_v = kv_cache.update(0, inp_k, inp_v)

    print(f"Returned Cache Key Shape after 3rd Update: {cached_k.shape}")
    print(f"Returned Cache Val Shape after 3rd Update: {cached_v.shape}")