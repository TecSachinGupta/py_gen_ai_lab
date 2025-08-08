"""
Mini Llama Model Implementation

A complete implementation of a small-scale Llama-style transformer model with:
- Rotary Position Embeddings (RoPE)
- RMSNorm normalization
- SwiGLU activation
- Grouped Query Attention (GQA)
- KV caching for efficient inference
- Complete training pipeline

Author: Assistant
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json
import time


@dataclass
class ModelConfig:
    """Configuration class for Mini Llama model
    
    Contains all hyperparameters and training settings for the model.
    """
    # Model architecture
    vocab_size: int = 32000              # Size of vocabulary
    dim: int = 512                       # Hidden dimension
    n_layers: int = 6                    # Number of transformer layers
    n_heads: int = 8                     # Number of attention heads
    n_kv_heads: Optional[int] = 4        # Number of key-value heads (for GQA)
    mlp_ratio: int = 4                   # Ratio of MLP hidden dim to model dim
    max_seq_len: int = 2048              # Maximum sequence length
    dropout: float = 0.1                 # Dropout probability
    
    # Training hyperparameters
    weight_decay: float = 0.01           # Weight decay for optimizer
    learning_rate: float = 3e-4          # Peak learning rate
    warmup_steps: int = 1000             # Number of warmup steps
    max_steps: int = 10000               # Maximum training steps
    batch_size: int = 8                  # Training batch size
    gradient_accumulation_steps: int = 4  # Steps to accumulate gradients
    eval_interval: int = 500             # Steps between evaluations
    save_interval: int = 1000            # Steps between checkpoints


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    RMSNorm is more stable than LayerNorm and is used throughout Llama models.
    It normalizes the input by the root mean square of the elements.
    
    Args:
        dim: The dimension of the input tensor
        eps: Small constant for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS normalization factor
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)
    
    RoPE encodes positional information by rotating the query and key vectors
    in the attention mechanism. This allows the model to better understand
    relative positions between tokens.
    
    Args:
        dim: Dimension of the embeddings (should be head_dim)
        max_seq_len: Maximum sequence length to precompute embeddings for
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        
        # Create frequency bands for rotation
        # Higher frequencies for later dimensions
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           -(math.log(10000.0) / dim))
        
        # Precompute sin and cos for all positions and frequencies
        # These will be used to rotate the embeddings
        self.register_buffer('cos_cached', torch.cos(position * div_term))
        self.register_buffer('sin_cached', torch.sin(position * div_term))

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin values for the given sequence length
        
        Args:
            x: Input tensor (used for device/dtype info)
            seq_len: Length of the sequence
            
        Returns:
            Tuple of (cos, sin) tensors for the sequence
        """
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor
    
    This function rotates pairs of dimensions in the input tensor according
    to their position using precomputed cos and sin values.
    
    Args:
        x: Input tensor of shape (..., seq_len, dim)
        cos: Cosine values for rotation
        sin: Sine values for rotation
        
    Returns:
        Rotated tensor of same shape as input
    """
    # Split x into pairs for rotation (even and odd dimensions)
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation using rotation matrix:
    # [cos -sin] [x1]
    # [sin  cos] [x2]
    rotated = torch.stack([
        x1 * cos - x2 * sin,  # New x1
        x1 * sin + x2 * cos   # New x2
    ], dim=-1)
    
    # Flatten the last two dimensions to restore original shape
    return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU activation function
    
    SwiGLU (Swish-Gated Linear Unit) is the activation function used in Llama.
    It combines a gating mechanism with the SiLU (Swish) activation function.
    
    The formula is: SwiGLU(x) = (W1(x) * SiLU(W3(x))) * W2
    where W1 is the gate projection, W3 is the up projection, and W2 is down projection.
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (usually 4 * dim in transformers)
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Down projection  
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Activated tensor of same shape as input
        """
        # Compute gate with SiLU activation
        gate = F.silu(self.w1(x))  # SiLU(W1(x))
        # Compute up projection
        up = self.w3(x)            # W3(x)
        # Combine gate and up projection, then down project
        return self.w2(gate * up)  # W2(gate * up)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention and KV caching
    
    This implements the attention mechanism used in Llama models with:
    - Grouped Query Attention (GQA) for efficiency
    - Rotary Position Embeddings (RoPE)
    - KV caching for fast inference
    - Causal masking for autoregressive generation
    
    Args:
        dim: Model dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key-value heads (for GQA, can be < n_heads)
        max_seq_len: Maximum sequence length for precomputing masks
    """
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads else n_heads
        self.head_dim = dim // n_heads
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        # Linear projections for queries, keys, values, and output
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # Rotary position embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
        
        # Precompute causal mask for autoregressive attention
        # Lower triangular matrix ensures we only attend to previous positions
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )

    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of multi-head attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            kv_cache: Optional cached key-value pairs from previous steps
            use_cache: Whether to return updated cache for next step
            
        Returns:
            Tuple of (attention_output, new_cache)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, and values
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Handle KV caching for efficient inference
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate new k,v with cached k,v
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        
        # Apply rotary position embeddings
        start_pos = k.shape[1] - seq_len if kv_cache is not None else 0
        cos, sin = self.rotary_emb(x, k.shape[1])
        
        # Apply RoPE to queries and keys
        q_rope = apply_rotary_pos_emb(q, cos[start_pos:start_pos+seq_len], sin[start_pos:start_pos+seq_len])
        k_rope = apply_rotary_pos_emb(k, cos[:k.shape[1]], sin[:k.shape[1]])
        
        # Transpose for attention computation: (batch, heads, seq_len, head_dim)
        q = q_rope.transpose(1, 2)
        k = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle Grouped Query Attention (GQA)
        # If we have fewer KV heads than query heads, repeat the KV heads
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask to prevent attending to future positions
        kv_seq_len = k.shape[2]
        if kv_cache is not None:
            # For cached inference, mask only applies to new tokens
            mask = self.mask[start_pos:start_pos+seq_len, :kv_seq_len]
        else:
            # For training, mask applies to full sequence
            mask = self.mask[:seq_len, :seq_len]
        
        # Set masked positions to negative infinity (will become 0 after softmax)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.wo(out)
        
        # Prepare cache for next iteration if requested
        new_cache = None
        if use_cache:
            new_cache = (k.transpose(1, 2), v.transpose(1, 2))
        
        return out, new_cache


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture
    
    Each block consists of:
    1. Multi-head attention with residual connection and pre-norm
    2. Feed-forward network with residual connection and pre-norm
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads for GQA
        mlp_ratio: Ratio of MLP hidden dimension to model dimension
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        n_kv_heads: Optional[int] = None,
        mlp_ratio: int = 4,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, n_kv_heads, max_seq_len)
        self.feed_forward = SwiGLU(dim, dim * mlp_ratio)
        
        # Pre-norm layers (applied before attention and MLP)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of transformer block
        
        Args:
            x: Input tensor
            kv_cache: Optional KV cache from previous step
            use_cache: Whether to return updated cache
            
        Returns:
            Tuple of (output, new_cache)
        """
        # Pre-norm attention with residual connection
        attn_out, new_cache = self.attention(
            self.attention_norm(x), kv_cache, use_cache
        )
        x = x + attn_out
        
        # Pre-norm feed-forward with residual connection
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x, new_cache


class MiniLlama(nn.Module):
    """Mini Llama transformer model
    
    A complete implementation of a Llama-style decoder-only transformer with:
    - Token embeddings
    - Multiple transformer blocks
    - Final layer normalization
    - Language modeling head
    
    Args:
        vocab_size: Size of the vocabulary
        dim: Model dimension (hidden size)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads for GQA
        mlp_ratio: Ratio of MLP hidden dim to model dim
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        mlp_ratio: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        
        # Token embeddings - convert token IDs to dense vectors
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, mlp_ratio, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(dim)
        
        # Output projection to vocabulary (language modeling head)
        # Note: No bias is used in Llama models
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using standard techniques
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights of the model
        
        Uses Xavier/Glorot normal initialization for linear layers
        and normal initialization for embeddings.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        tokens: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass of the model
        
        Args:
            tokens: Input token IDs of shape (batch_size, seq_len)
            kv_caches: Optional list of KV caches for each layer
            use_cache: Whether to return updated caches
            
        Returns:
            Tuple of (logits, new_caches)
            - logits: Output logits of shape (batch_size, seq_len, vocab_size)
            - new_caches: Updated KV caches for each layer (if use_cache=True)
        """
        # Convert tokens to embeddings
        x = self.tok_embeddings(tokens)
        x = self.dropout(x)
        
        # Initialize or use existing caches
        if use_cache and kv_caches is None:
            kv_caches = [None] * self.n_layers
        
        new_caches = [] if use_cache else None
        
        # Pass through each transformer layer
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)
        
        # Apply final normalization and output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, new_caches

    def generate(
        self, 
        tokens: torch.Tensor, 
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively
        
        Uses KV caching for efficient generation. Supports various sampling strategies.
        
        Args:
            tokens: Initial token sequence of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling (if specified)
            top_p: Nucleus sampling threshold (if specified)
            use_cache: Whether to use KV caching for efficiency
            
        Returns:
            Generated token sequence including input tokens
        """
        self.eval()
        generated = tokens.clone()
        kv_caches = None
        
        with torch.no_grad():
            for i in range(max_new_tokens):
                # For first iteration, use full sequence
                # For subsequent iterations, use only the last token (with caching)
                input_tokens = generated if i == 0 else generated[:, -1:]
                
                # Forward pass
                logits, kv_caches = self.forward(input_tokens, kv_caches, use_cache)
                
                # Get logits for the last token and apply temperature
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we reach maximum sequence length
                if generated.shape[1] >= self.layers[0].attention.mask.shape[0]:
                    break
        
        return generated

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate Model FLOPs Utilization (MFU)
        
        Estimates what percentage of the theoretical peak FLOPs of the hardware
        we're actually achieving during training.
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time taken per iteration
            
        Returns:
            MFU as a ratio (0.0 to 1.0+)
        """
        # Estimate FLOPs in forward pass
        # This is an approximation based on standard transformer FLOP counting
        N = sum(p.numel() for p in self.parameters())
        L, H, Q, T = self.n_layers, self.n_heads, self.dim//self.n_heads, 2048
        
        # 6N for the forward pass, 12LHQ*T for attention operations
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        # Calculate achieved FLOPs per second
        flops_achieved = flops_per_iter * (1.0/dt)
        
        # Compare to A100 bfloat16 peak performance
        flops_promised = 312e12  # 312 TFLOPS for A100
        mfu = flops_achieved / flops_promised
        return mfu


class TextDataset(Dataset):
    """Simple text dataset for language model training
    
    Takes a list of tokenized sequences and prepares them for autoregressive
    language modeling by creating input-target pairs.
    
    Args:
        data: List of tokenized sequences
        seq_len: Maximum sequence length
    """
    
    def __init__(self, data: List[List[int]], seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training example
        
        Returns:
            Tuple of (input_tokens, target_tokens)
            - input_tokens: tokens[:-1] 
            - target_tokens: tokens[1:] (shifted by one position)
        """
        tokens = self.data[idx]
        
        # Handle sequences longer than max length
        if len(tokens) > self.seq_len:
            # Random crop for variety
            start_idx = torch.randint(0, len(tokens) - self.seq_len + 1, (1,)).item()
            tokens = tokens[start_idx:start_idx + self.seq_len]
        else:
            # Pad shorter sequences
            tokens = tokens + [0] * (self.seq_len - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        # Input is all tokens except last, target is all tokens except first
        return tokens[:-1], tokens[1:]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Create cosine learning rate schedule with warmup
    
    Learning rate increases linearly during warmup, then decreases following
    a cosine curve. This is the standard schedule used in transformer training.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


class MiniLlamaTrainer:
    """Trainer class for Mini Llama model
    
    Handles the complete training pipeline including:
    - Optimizer setup with weight decay
    - Learning rate scheduling
    - Training loop with gradient accumulation
    - Evaluation and checkpointing
    - Loss logging and performance monitoring
    
    Args:
        config: Model configuration
        model: Mini Llama model instance
        device: Device to train on ('cuda' or 'cpu')
    """
    
    def __init__(self, config: ModelConfig, model: MiniLlama, device: str = 'cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Setup optimizer with weight decay
        # Apply weight decay to weights but not biases or norms
        decay_params = []
        nodecay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and normalization layers
                if any(nd in name for nd in ['bias', 'norm']):
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
        
        # Create optimizer with different weight decay for different parameter groups
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ], lr=config.learning_rate, betas=(0.9, 0.95))
        
        # Setup learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, config.warmup_steps, config.max_steps
        )
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')

    def save_checkpoint(self, path: str, loss: float):
        """Save model checkpoint
        
        Saves model state, optimizer state, scheduler state, and training metadata.
        
        Args:
            path: Path to save checkpoint
            loss: Current loss value
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'loss': loss,
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint
        
        Restores model state, optimizer state, scheduler state, and training step.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {path}, step {self.step}")

    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """Evaluate model on validation data
        
        Runs the model in evaluation mode and computes average loss
        over the validation dataset.
        
        Args:
            eval_dataloader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (input_ids, targets) in enumerate(eval_dataloader):
                # Limit evaluation to 100 batches for speed
                if batch_idx >= 100:
                    break
                    
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids)
                
                # Compute cross-entropy loss
                # Reshape logits and targets for loss computation
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    targets.reshape(-1), 
                    ignore_index=0  # Ignore padding tokens
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else float('inf')

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop
        
        Implements the complete training procedure with:
        - Gradient accumulation
        - Learning rate scheduling
        - Periodic evaluation and checkpointing
        - Loss logging and performance monitoring
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for validation data
        """
        self.model.train()
        running_loss = 0
        log_interval = 50  # Log every 50 steps
        
        print(f"Starting training for {self.config.max_steps} steps")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        # Main training loop
        while self.step < self.config.max_steps:
            for batch_idx, (input_ids, targets) in enumerate(train_dataloader):
                # Stop if we've reached max steps
                if self.step >= self.config.max_steps:
                    break
                
                # Move data to device
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), 
                    targets.reshape(-1), 
                    ignore_index=0  # Ignore padding tokens
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                running_loss += loss.item()
                
                # Update weights after accumulating gradients
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Logging
                    if self.step % log_interval == 0:
                        elapsed = time.time() - start_time
                        lr = self.scheduler.get_last_lr()[0]
                        
                        # Estimate model FLOPs utilization
                        mfu = self.model.estimate_mfu(
                            self.config.batch_size * self.config.gradient_accumulation_steps, 
                            elapsed / log_interval
                        )
                        
                        print(f"Step {self.step:5d} | Loss: {running_loss/log_interval:.4f} | "
                              f"LR: {lr:.2e} | MFU: {mfu*100:.2f}% | "
                              f"Time: {elapsed/log_interval:.2f}s/step")
                        
                        running_loss = 0
                        start_time = time.time()
                    
                    # Evaluation
                    if eval_dataloader and self.step % self.config.eval_interval == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        print(f"Step {self.step} | Eval Loss: {eval_loss:.4f}")
                        
                        # Save best model
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.save_checkpoint(f"checkpoints/best_model_step_{self.step}.pt", eval_loss)
                        
                        # Return to training mode
                        self.model.train()
                    
                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint(f"checkpoints/model_step_{self.step}.pt", running_loss)

        print(f"Training completed after {self.step} steps")


def prepare_dummy_data(vocab_size: int, num_sequences: int, max_seq_len: int) -> List[List[int]]:
    """Prepare dummy training data for demonstration
    
    In practice, you would replace this with your actual tokenized text data.
    
    Args:
        vocab_size: Size of vocabulary
        num_sequences: Number of sequences to generate
        max_seq_len: Maximum length of each sequence
        
    Returns:
        List of tokenized sequences
    """
    dummy_data = []
    for _ in range(num_sequences):
        # Generate random sequence length
        seq_len = torch.randint(50, max_seq_len, (1,)).item()
        # Generate random tokens (excluding 0 which is used for padding)
        tokens = torch.randint(1, vocab_size, (seq_len,)).tolist()
        dummy_data.append(tokens)
    
    return dummy_data


def create_data_loaders(data: List[List[int]], config: ModelConfig, train_split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders
    
    Args:
        data: List of tokenized sequences
        config: Model configuration
        train_split: Fraction of data to use for training
        
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    # Split data into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_data, config.max_seq_len)
    eval_dataset = TextDataset(eval_data, config.max_seq_len)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,  # Adjust based on your system
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_dataloader, eval_dataloader


def main():
    """Main function demonstrating model creation, training, and inference"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = ModelConfig(
        vocab_size=32000,
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=1024,
        batch_size=4,
        max_steps=5000,
        learning_rate=3e-4,
        warmup_steps=500,
        gradient_accumulation_steps=2
    )
    
    # Create model
    model = MiniLlama(
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare dummy data (replace with your actual data preparation)
    print("Preparing training data...")
    dummy_data = prepare_dummy_data(
        vocab_size=config.vocab_size,
        num_sequences=5000,  # Number of training sequences
        max_seq_len=config.max_seq_len
    )
    
    # Create data loaders
    train_dataloader, eval_dataloader = create_data_loaders(dummy_data, config)
    print(f"Training data: {len(train_dataloader)} batches")
    print(f"Validation data: {len(eval_dataloader)} batches")
    
    # Create trainer
    trainer = MiniLlamaTrainer(config, model, device)
    
    # Training
    print("\nStarting training...")
    trainer.train(train_dataloader, eval_dataloader)
    
    # Demonstration of generation
    print("\nTesting generation...")
    model.eval()
    
    # Create some test input
    test_tokens = torch.randint(1, config.vocab_size, (1, 10)).to(device)
    print(f"Input tokens: {test_tokens.squeeze().tolist()}")
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            test_tokens, 
            max_new_tokens=20, 
            temperature=0.8, 
            top_k=50,
            use_cache=True
        )
    
    print(f"Generated tokens: {generated.squeeze().tolist()}")
    print(f"New tokens only: {generated.squeeze()[10:].tolist()}")


# Example usage and training setup
if __name__ == "__main__":
    """
    Example usage of the Mini Llama implementation.
    
    This demonstrates:
    1. Model creation with custom configuration
    2. Data preparation and loading
    3. Training setup and execution
    4. Text generation with various sampling strategies
    """
    
    # Run the main training and inference demo
    main()
    
    # Additional examples of model usage
    
    print("\n" + "="*50)
    print("ADDITIONAL EXAMPLES")
    print("="*50)
    
    # Example 1: Creating different model sizes
    print("\n1. Different model configurations:")
    
    # Tiny model for testing
    tiny_config = ModelConfig(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=512
    )
    tiny_model = MiniLlama(**tiny_config.__dict__)
    print(f"Tiny model: {sum(p.numel() for p in tiny_model.parameters()):,} parameters")
    
    # Medium model
    medium_config = ModelConfig(
        vocab_size=32000,
        dim=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=8,
        max_seq_len=2048
    )
    medium_model = MiniLlama(**{k: v for k, v in medium_config.__dict__.items() 
                              if k in ['vocab_size', 'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'max_seq_len']})
    print(f"Medium model: {sum(p.numel() for p in medium_model.parameters()):,} parameters")
    
    # Example 2: Different generation strategies
    print("\n2. Different generation strategies:")
    test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    
    # Greedy generation (temperature=0 equivalent)
    with torch.no_grad():
        greedy = tiny_model.generate(test_input, max_new_tokens=10, temperature=0.01)
        print(f"Greedy: {greedy.squeeze().tolist()}")
        
        # High temperature (more random)
        random_gen = tiny_model.generate(test_input, max_new_tokens=10, temperature=2.0)
        print(f"Random (temp=2.0): {random_gen.squeeze().tolist()}")
        
        # Top-k sampling
        topk_gen = tiny_model.generate(test_input, max_new_tokens=10, temperature=1.0, top_k=10)
        print(f"Top-k (k=10): {topk_gen.squeeze().tolist()}")
        
        # Nucleus (top-p) sampling
        topp_gen = tiny_model.generate(test_input, max_new_tokens=10, temperature=1.0, top_p=0.9)
        print(f"Top-p (p=0.9): {topp_gen.squeeze().tolist()}")
    
    # Example 3: Checkpoint saving/loading
    print("\n3. Checkpoint operations:")
    
    # Save a model
    checkpoint_path = "example_checkpoint.pt"
    dummy_trainer = MiniLlamaTrainer(tiny_config, tiny_model, 'cpu')
    dummy_trainer.save_checkpoint(checkpoint_path, 1.0)
    
    # Create new model and load checkpoint
    new_model = MiniLlama(**{k: v for k, v in tiny_config.__dict__.items() 
                           if k in ['vocab_size', 'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'max_seq_len']})
    new_trainer = MiniLlamaTrainer(tiny_config, new_model, 'cpu')
    new_trainer.load_checkpoint(checkpoint_path)
    
    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("\nMini Llama implementation complete!")
    print("\nTo use with real data:")
    print("1. Replace prepare_dummy_data() with your tokenization pipeline")
    print("2. Use a proper tokenizer (e.g., SentencePiece)")
    print("3. Adjust model size based on your computational resources")
    print("4. Monitor training loss and adjust hyperparameters")
    print("5. Implement proper evaluation metrics for your task")