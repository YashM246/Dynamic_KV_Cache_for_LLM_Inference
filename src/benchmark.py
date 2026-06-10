import torch
from attention import MultiHeadAttention
from attention import generate
from static_kv_cache import generate_with_static_cache
from dynamic_kv_cache import generate_with_dynamic_cache
from paged_attention import paged_generate
torch.set_float32_matmul_precision('high')
import pandas as pd

BATCH_SIZE = 1
SEQ_LEN = 512
MAX_SEQ_LEN = 1024
D_MODEL = 512
NUM_HEADS = 16
HEAD_DIM = 256
NUM_LAYERS = 1
STEPS = 512
WARMUP_RUNS = 10

benchmark = {}

print("\n#################")
print("#  CPU Testing  #")
print("#################\n")

# Random Input Sequence
inp_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

# Instantiate Models
model = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
compiled_model = torch.compile(model, dynamic=True)

# Normal Model Run
output, times = generate(model=model, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n###############################")
print("#  Normal Model Run No Cache  #")
print("###############################\n")
print(f"Total Prefill Time: NA")
print(f"Mean Decode Step Time: {(sum(times)/len(times))} ms")
print(f"Max Step Time: {max(times)} ms")
print(f"Min Step Time: {min(times)} ms")
print(f"Total time for {STEPS} steps/generations: {sum(times)} ms")

benchmark["model_cpu_no_cache"] = {"Prefill Time (ms)":"NA", 
                                   "Mean Decode Time (ms)":(sum(times)/len(times)),
                                   "Max Decode Time (ms)":max(times),
                                   "Min Decode Time (ms)":min(times),
                                   "Total Time (Prefill + Decode) (ms)":sum(times),
                                   }

# Compiled Model Run
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))
_, _ = generate(compiled_model, warmup_seq, WARMUP_RUNS)

output, times = generate(model=compiled_model, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n###############################")
print("#  Compiled Model Run No Cache  #")
print("###############################\n")
print(f"Total Prefill Time: NA")
print(f"Mean Decode Step Time: {(sum(times)/len(times))} ms")
print(f"Max Step Time: {max(times)} ms")
print(f"Min Step Time: {min(times)} ms")
print(f"Total time for {STEPS} steps/generations: {sum(times)} ms")

benchmark["compiled_model_cpu_no_cache"] = {"Prefill Time (ms)":"NA", 
                                   "Mean Decode Time (ms)":(sum(times)/len(times)),
                                   "Max Decode Time (ms)":max(times),
                                   "Min Decode Time (ms)":min(times),
                                   "Total Time (Prefill + Decode) (ms)":sum(times),
                                   }

# Normal Model + Static KV Cache
output, prefill_time, decode_times = generate_with_static_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

print("\n######################################")
print("#  Normal Model Run Static KV Cache  #")
print("######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_cpu_static_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Static KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))
_, _, _ = generate_with_static_cache(compiled_model, NUM_LAYERS, warmup_seq, MAX_SEQ_LEN, WARMUP_RUNS)

output, prefill_time, decode_times = generate_with_static_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

print("\n########################################")
print("#  Compiled Model Run Static KV Cache  #")
print("########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_cpu_static_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Normal Model + Dynamic KV Cache
output, prefill_time, decode_times = generate_with_dynamic_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#######################################")
print("#  Normal Model Run Dynamic KV Cache  #")
print("#######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_cpu_dynamic_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Dynamic KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))
_, _, _ = generate_with_dynamic_cache(compiled_model, NUM_LAYERS, warmup_seq, WARMUP_RUNS)

output, prefill_time, decode_times = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#########################################")
print("#  Compiled Model Run Dynamic KV Cache  #")
print("#########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_cpu_dynamic_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Normal Model + Paged Cache
output, prefill_time, decode_times = paged_generate(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#######################################")
print("#  Normal Model Run Paged KV Cache  #")
print("#######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_cpu_paged_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Paged KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))
_, _, _ = paged_generate(compiled_model, NUM_LAYERS, warmup_seq, WARMUP_RUNS)

output, prefill_time, decode_times = paged_generate(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#########################################")
print("#  Compiled Model Run Paged KV Cache  #")
print("#########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_cpu_paged_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }




print("\n#################")
print("#  GPU Testing  #")
print("#################\n")
torch.cuda.empty_cache()

# Random Input Sequence
inp_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()

# Instantiate Models
model = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS).cuda()
compiled_model = torch.compile(model, dynamic=True).cuda()

# Normal Model Run
output, times = generate(model=model, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n###############################")
print("#  Normal Model Run No Cache  #")
print("###############################\n")
print(f"Total Prefill Time: NA")
print(f"Mean Decode Step Time: {(sum(times)/len(times))} ms")
print(f"Max Step Time: {max(times)} ms")
print(f"Min Step Time: {min(times)} ms")
print(f"Total time for {STEPS} steps/generations: {sum(times)} ms")

benchmark["model_gpu_no_cache"] = {"Prefill Time (ms)":"NA", 
                                   "Mean Decode Time (ms)":(sum(times)/len(times)),
                                   "Max Decode Time (ms)":max(times),
                                   "Min Decode Time (ms)":min(times),
                                   "Total Time (Prefill + Decode) (ms)":sum(times),
                                   }

# Compiled Model Run
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()
_, _ = generate(compiled_model, warmup_seq, WARMUP_RUNS)

output, times = generate(model=compiled_model, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n###############################")
print("#  Compiled Model Run No Cache  #")
print("###############################\n")
print(f"Total Prefill Time: NA")
print(f"Mean Decode Step Time: {(sum(times)/len(times))} ms")
print(f"Max Step Time: {max(times)} ms")
print(f"Min Step Time: {min(times)} ms")
print(f"Total time for {STEPS} steps/generations: {sum(times)} ms")

benchmark["compiled_model_gpu_no_cache"] = {"Prefill Time (ms)":"NA", 
                                   "Mean Decode Time (ms)":(sum(times)/len(times)),
                                   "Max Decode Time (ms)":max(times),
                                   "Min Decode Time (ms)":min(times),
                                   "Total Time (Prefill + Decode) (ms)":sum(times),
                                   }

# Normal Model + Static KV Cache
output, prefill_time, decode_times = generate_with_static_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

print("\n######################################")
print("#  Normal Model Run Static KV Cache  #")
print("######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_gpu_static_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Static KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()
_, _, _ = generate_with_static_cache(compiled_model, NUM_LAYERS, warmup_seq, MAX_SEQ_LEN, WARMUP_RUNS)

output, prefill_time, decode_times = generate_with_static_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), max_seq_len=MAX_SEQ_LEN, num_steps=STEPS)

print("\n########################################")
print("#  Compiled Model Run Static KV Cache  #")
print("########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_gpu_static_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Normal Model + Dynamic KV Cache
output, prefill_time, decode_times = generate_with_dynamic_cache(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#######################################")
print("#  Normal Model Run Dynamic KV Cache  #")
print("#######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_gpu_dynamic_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Dynamic KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()
_, _, _ = generate_with_dynamic_cache(compiled_model, NUM_LAYERS, warmup_seq, WARMUP_RUNS)

output, prefill_time, decode_times = generate_with_dynamic_cache(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#########################################")
print("#  Compiled Model Run Dynamic KV Cache  #")
print("#########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_gpu_dynamic_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Normal Model + Paged Cache
output, prefill_time, decode_times = paged_generate(model=model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#######################################")
print("#  Normal Model Run Paged KV Cache  #")
print("#######################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["model_gpu_paged_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }

# Compiled Model + Paged KV Cache
warmup_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL)).cuda()
_, _, _ = paged_generate(compiled_model, NUM_LAYERS, warmup_seq, WARMUP_RUNS)

output, prefill_time, decode_times = paged_generate(model=compiled_model, num_layers=NUM_LAYERS, input_seq=inp_seq.clone(), num_steps=STEPS)

print("\n#########################################")
print("#  Compiled Model Run Paged KV Cache  #")
print("#########################################\n")
print(f"Total Prefill Time: {prefill_time} ms")
print(f"Mean Decode Step Time: {(sum(decode_times)/len(decode_times))} ms")
print(f"Max Step Time: {max(decode_times)} ms")
print(f"Min Step Time: {min(decode_times)} ms")
print(f"Total time for {STEPS} steps/generations (Prefill + {STEPS-1} Decode): {prefill_time + sum(decode_times)} ms")

benchmark["compiled_model_gpu_paged_kv_cache"] = {"Prefill Time (ms)":prefill_time, 
                                   "Mean Decode Time (ms)":(sum(decode_times)/len(decode_times)),
                                   "Max Decode Time (ms)":max(decode_times),
                                   "Min Decode Time (ms)":min(decode_times),
                                   "Total Time (Prefill + Decode) (ms)":prefill_time + sum(decode_times),
                                   }


pd.DataFrame(benchmark).T.to_csv("results/benchmark_results.csv")