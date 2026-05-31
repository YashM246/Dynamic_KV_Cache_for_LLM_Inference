import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        """
            Args:
                d_model -> Total dimension of input embeddings
                num_heads -> Number of parallel attention heads
        """
        
        super().__init__()
        assert d_model%num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.d_model = d_model
        self.num_heads = num_heads
        # d_k -> Dimensions per Head
        self.d_k = d_model // num_heads

        # Projections for Q, K, V vectors
        # Instead of making individual num_head layers, we will project
        # everything into a single large matrix (d_model -> d_model) and
        # split it later

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Output projection Layer to mix information from all heads back together
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, layer_idx=0, cache=None):
        """
            Args:
                q -> Query Matrix of shape (batch_size, seq_len_q, d_model)
                k -> Query Matrix of shape (batch_size, seq_len_k, d_model)
                v -> Query Matrix of shape (batch_size, seq_len_v, d_model)
                mask -> Optional tensor to hide future tokens
                layer_idx -> Optional layer_idx parameter for KV Cache Updating
                cache -> Optional parameter to pass cache
        """

        batch_size = q.size(0)

        ###########################################
        #  Step 1: Linear Proj. & Head Splitting  #
        ###########################################

        # Project tokens into Q, K, V spaces
        # Shape change from (batch_size, seq_len, d_model) to (batch_size, seq_len, d_model)
        q_proj = self.w_q(q)
        k_proj = self.w_k(k)
        v_proj = self.w_v(v)

        # Split d_model into (num_heads, d_k) and swap axis 1 and 2 (Transpose)
        # We transpose so that "num_heads" becomes the second_dimension. This lets
        # us compute attention for all heads in parallel using standard batch 
        # matrix multiplication
        # Shape Progression:
        #       view() -> (batch_size, seq_len, num_heads, d_k)
        #       transpose() -> (batch_size, num_heads, seq_len, d_k)
        q_heads = q_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = k_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v_proj.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Check if KV Cache is being used
        if cache is not None:
            # Squeeze k_heads and v_heads from (batch_size, num_heads, seq_len, d_k) to (batch_size, num_heads, head_dim)
            # d_k = head_dim (interchangeable)
            k_heads = k_heads[:, :, 0, :]
            v_heads = v_heads[:, :, 0, :]

            k_heads, v_heads = cache.update(layer_idx, k_heads, v_heads)

        ##########################################
        #  Step 2: Scaled Dot Product Attention  #
        ##########################################

        # Computing similarity between Q and K
        # We transpose last 2 dimensions of k_heads to align the shapes
        # q_heads: (batch_size, num_heads, seq_len_q, d_k)
        # k_heads.T: (batch_size, num_heads, d_k, seq_len_k)
        # Result: (batch_size, num_heads, seq_len_q, seq_len_k)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))

        # Scale by sq.root of d_k
        # Prevents scores from going too large (no vanishing softmax gradients)
        scores = scores / (self.d_k ** 0.5)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask==True, float("-inf"))

        # Convert raw scores to probabilities along key seq len dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Multiply atten prob by Value vectors
        # Shape: (batch_size, num_heads, seq_len_q, d_k)
        attn_output = torch.matmul(attn_weights, v_heads)


        #################################################
        #  Step 3: Concatenation and Output Projection  #
        #################################################

        # Swap num_heads and seq_len back to their original spots
        # Shape: (batch_size, seq_len_q, num_heads, d_k)
        attn_output = attn_output.transpose(1, 2)

        # Merge all parallel heads back to a single vector of size d_model
        # contiguous() ensures memory layout is unbroken before calling view()
        # Shape: (batch_size, seq_len_q, d_model)
        attn_output = attn_output.contiguous().view(batch_size, -1, self.d_model)

        # Run through final linear layer to allow heads to interact with each other
        return self.w_o(attn_output)
    
def make_causal_mask(seq_len, device):
    """
        To create a boolean causal mask where 'True" means block
    """

    # Create Basic Tensor of shape seq_len, seq_len
    base_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)

    # Extract upper triangle (diagonal=1 to exclude main diagonal)
    # This will keep True in Upper Right and False everywhere else
    causal_mask = torch.triu(base_mask, diagonal=1)
    return causal_mask

# torch.no_grad() since we are not training
# No need to waste time and computation in calculating gradients
@torch.no_grad()    
def generate(model, input_seq, num_steps):
    device = input_seq.device
    times = []

    for _ in range(num_steps):

        start_time = time.perf_counter()

        # Extract batch size, seq_len, and model dimension from input
        _, seq_len, _ = input_seq.shape

        # Use seq_len to create a mask for current length
        mask = make_causal_mask(seq_len, device)

        # call model, get self attention scores
        output = model(input_seq, input_seq, input_seq, mask)

        # Take last token's output and
        # concatenate it into input_seq alone dim=1
        input_seq = torch.cat((input_seq, output[:, -1:, :]), dim=1)
        # output[:, -1:, :] -> to keep seq dimension in shape (batch, 1, d_model)

        end_time = time.perf_counter()

        times.append((end_time-start_time)*1000)

    return times

if __name__ == "__main__":

    BATCH_SIZE = 1
    SEQ_LEN = 10
    D_MODEL = 256

    # Create a random input_seq
    input_seq = torch.randn((BATCH_SIZE, SEQ_LEN, D_MODEL))

    # Instantiate Model
    model = MultiHeadAttention(d_model= D_MODEL, num_heads= 4)

    # Run Generate Function
    times = generate(model=model, input_seq=input_seq, num_steps=50)

    print(f"Mean Step Time: {(sum(times)/len(times))} ms")
    print(f"Max Step Time: {max(times)} ms")
    print(f"Min Step Time: {min(times)} ms")