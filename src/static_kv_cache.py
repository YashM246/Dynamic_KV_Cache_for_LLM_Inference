import torch
import torch.nn as nn
import time
from attention import MultiHeadAttention
from attention import generate

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

@torch.no_grad()
def generate_with_static_cache(model, num_layers, input_seq, max_seq_len, num_steps):

    # Prefill Phase

    start_time = time.perf_counter()

    batch_size, seq_len, d_model = input_seq.shape
    num_heads = model.num_heads
    head_dim = model.d_k
    cache = Static_KV_Cache(num_layers, batch_size, num_heads, max_seq_len, head_dim)

    for i in range(seq_len):
        output = model(input_seq[:, i:i+1, :], input_seq[:, i:i+1, :], input_seq[:, i:i+1, :], mask=None, cache=cache)
    
    input_seq = torch.cat((input_seq, output[:, -1:, :]), dim=1)
    
    end_time = time.perf_counter()

    prefill_time = ((end_time-start_time)*1000)

    # Decode Phase
    
    decode_times = []

    for _ in range(num_steps-1):
        start_time = time.perf_counter()
        output = model(input_seq[:, -1:, :], input_seq[:, -1:, :], input_seq[:, -1:, :], mask=None, cache=cache)
        input_seq = torch.cat((input_seq, output[:, -1:, :]), dim=1)
        end_time = time.perf_counter()
        decode_times.append((end_time-start_time)*1000)

    return input_seq, prefill_time, decode_times

if __name__ == "__main__":

    NUM_LAYERS = 1
    BATCH_SIZE = 4
    NUM_HEADS = 8
    MAX_SEQ_LEN = 128
    HEAD_DIM = 256

    print("#########################")
    print("#  KV Cache Validation  #")
    print("#########################\n")

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

    print("\n\n####################################")
    print("#  KV Cache Generation Validation  #")
    print("####################################\n")

    BATCH_SIZE = 1
    SEQ_LEN = 10
    MAX_SEQ_LEN = 100
    D_MODEL = 256
    NUM_LAYERS = 1
    STEPS = 50

    # Create a random input_seq
    input_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

    # Instantiate Model
    model = MultiHeadAttention(d_model= D_MODEL, num_heads= 4)

    # Run Generate Function
    _, prefill_time, decode_times = generate_with_static_cache(model=model, num_layers=NUM_LAYERS, input_seq=input_seq, max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

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

    output_cached, _, _ = generate_with_static_cache(model=model, num_layers=NUM_LAYERS, input_seq=input_seq.clone(), max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

    if torch.allclose(output_cached, output_no_cache, atol=1e-5):
        print("PASS: Outputs Match\n")
    else:
        print("FAIL: Outputs differ\n")
