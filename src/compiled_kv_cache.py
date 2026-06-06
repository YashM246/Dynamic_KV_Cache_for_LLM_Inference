from dynamic_kv_cache import Dynamic_KV_Cache
from dynamic_kv_cache import generate_with_dynamic_cache
import torch
import triton
from attention import MultiHeadAttention
torch.set_float32_matmul_precision('high')

BATCH_SIZE = 4
SEQ_LEN = 100
D_MODEL = 1024
NUM_HEADS = 64
HEAD_DIM = 256
NUM_LAYERS = 1
STEPS = 100
WARMUP_RUNS = 10

print("\n#################")
print("#  CPU Testing  #")
print("#################\n")

# Random Input Sequence
inp_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

# Instantiate Models
model = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
compiled_model = torch.compile(model, dynamic=True)

# Normal Model Run
output, prefill_time, decode_times = generate_with_dynamic_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq, num_steps=STEPS)

print("\n######################")
print("#  Normal Model Run  #")
print("######################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

# Compiled Function Run

# Warmup
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))
_, _, _ = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=warmup_seq, num_steps=WARMUP_RUNS)

# Compiled Model Run
output, prefill_time, decode_times = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq, num_steps=STEPS)

print("\n########################")
print("#  Compiled Model Run  #")
print("########################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")


print("\n##############")
print("#  GPU Test  #")
print("##############\n")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(triton.__version__)

# Random Input Sequence
inp_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()

# Instantiate Models
model = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS).cuda()
compiled_model = torch.compile(model, dynamic=True).cuda()

# Normal Model Run
output, prefill_time, decode_times = generate_with_dynamic_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq, num_steps=STEPS)

print("\n######################")
print("#  Normal Model Run  #")
print("######################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

# Compiled Function Run

# Warmup
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()
_, _, _ = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=warmup_seq, num_steps=WARMUP_RUNS)

# Compiled Model Run
output, prefill_time, decode_times = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq, num_steps=STEPS)

print("\n########################")
print("#  Compiled Model Run  #")
print("########################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")