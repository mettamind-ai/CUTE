# Dynamic Chunking End2End HNet https://arxiv.org/html/2507.07955v1
# NOTE: This is an experimental implementation attempting end-to-end language modeling
# without explicit tokenization. Key differences from H-Net paper:
# - Uses fixed 8 tokens instead of learned dynamic chunking
# - Prime-based signal encoding instead of learned compression
# - Single-stage hierarchy vs. multi-stage in paper
# - Character-level vs. byte-level input

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt

# =============================================================================
# Utility function to generate the first n prime numbers (naively)
# NOTE: Used for signal encoding - heuristic approach instead of learned frequencies
# Could be replaced with learned embedding table per H-Net recommendations
# =============================================================================
def generate_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
        num += 1
    return primes

# =============================================================================
# Dataset for training: each sample is a sliding window from the text.
# NOTE: Character-level processing aligns with H-Net's byte-level approach
# but may be less efficient for non-English languages
# =============================================================================
class TextDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx):
        self.text = text  # NOTE: Full text corpus - H-Net uses streaming approach
        self.seq_length = seq_length  # total length = context length + 1 (target)
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.text) - self.seq_length + 1
    
    def __getitem__(self, idx):
        sample = self.text[idx:idx+self.seq_length]  # NOTE: Fixed window sliding - H-Net uses dynamic boundaries
        input_seq = sample[:-1]   # context characters - sequential processing vs. H-Net's chunking
        target_char = sample[-1]    # target character - single char prediction vs. H-Net's multi-step
        input_indices = torch.tensor([self.char_to_idx[c] for c in input_seq], dtype=torch.long)
        target_index = torch.tensor(self.char_to_idx[target_char], dtype=torch.long)
        return input_indices, target_index

# =============================================================================
# PatternEncoder: Integrates signal encoding, tokenization, and token refinement.
# =============================================================================
class PatternEncoder(nn.Module):
    """
    PatternEncoder:
    ------------------
    H-NET ARCHITECTURE IMPLEMENTATION NOTE:
    This implements a fixed-token approach (m_tokens=8) instead of H-Net's learned dynamic chunking.
    
    Key components:
    1. Signal Encoding: Uses prime-based frequencies (heuristic) vs. learned compression in H-Net
    2. Tokenization: Soft combination via weighted embeddings (similar to H-Net concept)
    3. Token Refinement: Self-attention between fixed 8 tokens (H-Net uses variable-length chunks)
    
    LIMITATIONS vs. H-Net paper:
    - Fixed 8 tokens instead of learned chunk boundaries
    - Single-stage compression vs. multi-stage hierarchy
    - Prime frequencies may not be optimal for all languages
    """
    
    # IMPLEMENTATION NOTES:
    # - signal_length: 512 points for frequency domain representation
    # - m_tokens: FIXED at 8 (should be dynamic per H-Net)
    # - token_vocab_size: 256 (hyperparameter for soft tokenization)
    def __init__(self, signal_length, m_tokens, token_vocab_size, token_embedding_dim, dropout_prob, device):
        super(PatternEncoder, self).__init__()
        self.signal_length = signal_length  # 512: Fixed frequency resolution - H-Net uses learned compression
        self.m_tokens = m_tokens            # 8: FIXED token count - H-Net uses dynamic chunking
        self.token_vocab_size = token_vocab_size  # 256: Soft tokenization vocab size
        self.token_embedding_dim = token_embedding_dim  # 64: Token representation dimension
        self.device = device
        
        # Signal Encoder: Create time vector for sinusoidal encoding
        # NOTE: Uses fixed linear spacing [0,1] - H-Net learns compression functions
        t = torch.linspace(0, 1, steps=signal_length, device=device)
        self.register_buffer("time_vector", t)  # Shape: (512,)
        
        # Tokenizer: Projects signal to token logits
        # NOTE: Fixed linear projection - H-Net uses learned chunking boundaries
        self.linear = nn.Linear(signal_length, m_tokens * token_vocab_size)  # 512 → 8×256
        
        # Soft tokenization via embedding lookup
        # NOTE: Weighted combination of embeddings - similar to H-Net concept
        self.token_embedding = nn.Embedding(token_vocab_size, token_embedding_dim)  # 256×64
        
        # Self-attention within fixed 8 tokens
        # NOTE: Processes 8 tokens regardless of input length - H-Net uses variable chunks
        self.token_attn = nn.MultiheadAttention(embed_dim=token_embedding_dim, num_heads=4, batch_first=True)
        self.token_dropout = nn.Dropout(0.1)  # Regularization during training
        self.token_norm = nn.LayerNorm(token_embedding_dim)  # Normalization for stability
        
        # TokenRefiner: Additional self-attention on 8 tokens
        # NOTE: Second attention layer - H-Net uses hierarchical processing
        self.refiner_attn = nn.MultiheadAttention(embed_dim=token_embedding_dim, num_heads=4, batch_first=True)
        self.refiner_dropout = nn.Dropout(dropout_prob)
        self.refiner_norm = nn.LayerNorm(token_embedding_dim)
        
    def forward(self, input_indices, prime_tensor):
        # Signal Encoding:
        # NOTE: This uses prime-based frequencies as heuristic signal encoding
        # H-Net paper recommends learned compression functions instead
        # input_indices: (batch, seq_length) - character indices
        # prime_tensor: (vocab_size,) - prime numbers for each character
        
        freqs = prime_tensor[input_indices]  # (batch, seq_length) - frequency for each char
        t = self.time_vector.unsqueeze(0).unsqueeze(0)  # (1,1,signal_length) - time base [0,1]
        freqs_expanded = freqs.unsqueeze(-1)  # (batch, seq_length, 1) - add frequency dimension
        
        # Generate sinusoidal signals for each character
        # NOTE: Uses sin(2πft) where f is character's prime - heuristic vs. learned
        signals = torch.sin(2 * math.pi * freqs_expanded * t)  # (batch, seq_length, 512)
        
        # Positional shifting and summation
        # NOTE: Cyclic shifts by position - could be replaced with learned positional encoding
        shifted_signals = []
        seq_length = input_indices.size(1)
        for i in range(seq_length):
            signal_i = signals[:, i, :]  # (batch, 512) - signal for char at position i
            shifted = torch.roll(signal_i, shifts=i, dims=1)  # Shift by position index
            shifted_signals.append(shifted)
        
        # Sum all shifted signals to create context representation
        # NOTE: Simple summation - H-Net uses learned pooling/compression
        context_signal = torch.stack(shifted_signals, dim=1).sum(dim=1)  # (batch, 512)
        
        # L2 normalization for stability
        # NOTE: Fixed normalization - H-Net learns appropriate scaling
        norm = context_signal.norm(p=2, dim=1, keepdim=True) + 1e-8
        normalized_signal = context_signal / norm  # (batch, 512)
        
        # Tokenization:
        # NOTE: Projects 512-dim signal to 8×256 logits - FIXED compression ratio
        # H-Net learns dynamic chunk boundaries instead of fixed projection
        batch_size = normalized_signal.size(0)
        logits = self.linear(normalized_signal)  # (batch, 8×256) = (batch, 2048)
        logits = logits.view(batch_size, self.m_tokens, self.token_vocab_size)  # (batch, 8, 256)
        
        # Soft tokenization via weighted embedding combination
        # NOTE: Similar concept to H-Net's soft chunking but with fixed 8 tokens
        probs = F.softmax(logits, dim=-1)  # (batch, 8, 256) - soft token assignments
        token_embeds = torch.matmul(probs, self.token_embedding.weight)  # (batch, 8, 64)
        
        # Self-attention within 8 fixed tokens
        # NOTE: Processes all 8 tokens regardless of input length - H-Net uses variable chunks
        attn_out, _ = self.token_attn(token_embeds, token_embeds, token_embeds)  # (batch, 8, 64)
        attn_out = self.token_dropout(attn_out)  # Regularization during training
        token_embeds = self.token_norm(token_embeds + attn_out)  # Residual connection
        
        # Token Refinement: Second self-attention layer
        # NOTE: Additional processing - H-Net uses hierarchical stages instead
        attn_out2, _ = self.refiner_attn(token_embeds, token_embeds, token_embeds)  # (batch, 8, 64)
        attn_out2 = self.refiner_dropout(attn_out2)
        refined = self.refiner_norm(token_embeds + attn_out2)  # (batch, 8, 64)
        
        # Flatten to 512-dim representation (8×64)
        # NOTE: Fixed flattening - H-Net maintains chunk structure for hierarchy
        token_flat = refined.reshape(batch_size, -1)  # (batch, 512)
        
        return token_flat, logits, token_embeds, normalized_signal

# =============================================================================
# PatternDecoder: Projects token representations to learned patterns and decodes them into characters.
# =============================================================================
class PatternDecoder(nn.Module):
    """
    PatternDecoder:
    ------------------
    H-NET DECODER IMPLEMENTATION NOTE:
    This implements the "pattern space → character logits" mapping.
    
    Key differences from H-Net paper:
    - Single-stage decoder vs. H-Net's multi-stage hierarchy
    - Fixed 8-token input vs. variable-length chunks
    - Character-level output vs. byte-level (could be extended)
    
    Architecture:
    1. Linear projection: 8×64 → 256 (pattern space)
    2. Self-attention refinement (single-head attention variant)
    3. Final linear: 256 → char_vocab_size
    
    LIMITATIONS:
    - No hierarchical refinement (H-Net uses multiple stages)
    - Fixed receptive field from 8 tokens
    """
    def __init__(self, m_tokens, token_embedding_dim, token_vocab_size, char_vocab_size):
        super(PatternDecoder, self).__init__()
        # Project flattened 8-token representation (8×64=512) to 256-dim pattern space
        # NOTE: Fixed projection - H-Net uses learned pattern spaces per hierarchical stage
        self.pattern_prediction = nn.Linear(m_tokens * token_embedding_dim, token_vocab_size)  # 512 → 256
        
        # Pre-decoder attention: Refines patterns within fixed space
        # NOTE: Single attention layer - H-Net uses multiple refinement stages
        self.pre_decoder_attn = nn.MultiheadAttention(embed_dim=token_vocab_size, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.1)  # Regularization during training
        self.norm = nn.LayerNorm(token_vocab_size)  # Stability normalization
        
        # Final decoder: Maps 256-dim patterns to character logits
        # NOTE: Direct character prediction - H-Net uses byte-level output
        # Could be extended to byte-level for better efficiency
        self.char_decoder = nn.Linear(token_vocab_size, char_vocab_size)  # 256 → char_vocab_size
        
    def forward(self, token_flat):
        # Decoding process: 8-token representation → character logits
        # NOTE: Single-stage decoding - H-Net uses hierarchical reconstruction
        # token_flat: (batch, 512) - flattened 8×64 token representation
        
        pattern_logits = self.pattern_prediction(token_flat)  # (batch, 256) - pattern space
        pattern_logits_seq = pattern_logits.unsqueeze(1)  # (batch, 1, 256) - add sequence dim
        
        # Self-attention refinement within pattern space
        # NOTE: Single attention layer - H-Net uses multiple refinement stages
        attn_out, _ = self.pre_decoder_attn(pattern_logits_seq, pattern_logits_seq, pattern_logits_seq)
        attn_out = self.dropout(attn_out)  # Regularization
        refined = self.norm(pattern_logits_seq + attn_out).squeeze(1)  # (batch, 256)
        
        # Final character prediction
        # NOTE: Direct character logits - H-Net uses byte-level prediction
        char_logits = self.char_decoder(refined)  # (batch, char_vocab_size)
        return char_logits

    def decode_debug(self, token_flat):
        """
        Runs the full decoding process and returns intermediate outputs as a dictionary.
        """
        outputs = {}
        pattern_logits = self.pattern_prediction(token_flat)
        outputs["pattern_logits"] = pattern_logits

        pattern_logits_seq = pattern_logits.unsqueeze(1)  # (batch, 1, token_vocab_size)
        attn_out, pre_decoder_attn_weights = self.pre_decoder_attn(pattern_logits_seq, pattern_logits_seq, pattern_logits_seq)
        attn_out = self.dropout(attn_out)
        refined = self.norm(pattern_logits_seq + attn_out).squeeze(1)  # (batch, token_vocab_size)
        outputs["refined_pattern"] = refined
        outputs["pre_decoder_attn_weights"] = pre_decoder_attn_weights

        char_logits = self.char_decoder(refined)
        outputs["char_logits"] = char_logits

        return outputs

# =============================================================================
# IntermediateTransformer: Applies a series of transformer encoder layers on token embeddings.
# NOTE: This is an ADDITIONAL component not present in H-Net paper
# H-Net uses multi-stage hierarchy for compression, this adds extra processing
# Could be removed to match paper architecture more closely
# =============================================================================
class IntermediateTransformer(nn.Module):
    """
    IntermediateTransformer:
    ---------------------------
    This block applies a series of transformer encoder layers (using PyTorch's TransformerEncoder)
    to refine the token sequence obtained from the PatternEncoder.

    The input is expected to have the shape: [batch, m_tokens, token_embedding_dim].
    The output is also of the same shape.
    """
    def __init__(self, token_embedding_dim, num_layers=2, num_heads=4, dropout=0.1):
        super(IntermediateTransformer, self).__init__()
        # NOTE: Additional transformer layers NOT in H-Net paper
        # H-Net uses multi-stage hierarchy instead of extra processing on fixed 8 tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_embedding_dim, nhead=num_heads, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 2 layers on 8 fixed tokens

    def forward(self, x):
        # NOTE: Processes fixed 8 tokens through transformer layers
        # H-Net would use multi-stage processing instead of this additional layer
        # x: (batch, 8, 64) - fixed token representation
        
        # Transformer expects (sequence, batch, embedding) format
        x = x.transpose(0,1)  # (8, batch, 64) - sequence dimension first
        x = self.transformer_encoder(x)  # (8, batch, 64) - 2 transformer layers
        x = x.transpose(0,1)  # (batch, 8, 64) - back to batch-first
        
        # NOTE: Still 8 tokens - no compression/expansion like H-Net's hierarchy
        return x

# =============================================================================
# End2EndModel: Composes the PatternEncoder and PatternDecoder to form the full model.
# =============================================================================
class End2EndModel(nn.Module):
    """
    End2EndModel:
    -----------------
    MAIN H-NET IMPLEMENTATION:
    
    This implements a single-stage end-to-end model with the following flow:
    Characters → Signal Encoding → 8 Tokens → Character Prediction
    
    DEVIATIONS FROM H-NET PAPER:
    1. Fixed 8 tokens vs. learned dynamic chunking boundaries
    2. Prime-based signal encoding vs. learned compression functions  
    3. Single-stage vs. multi-stage hierarchy
    4. Character-level vs. byte-level processing
    
    INPUT: Character indices (batch, seq_length)
    OUTPUT: Next character logits (batch, char_vocab_size)
    
    Architecture:
    1. PatternEncoder: Character → 8-token representation
    2. IntermediateTransformer: Additional processing (not in paper)
    3. PatternDecoder: 8 tokens → character logits
    
    RECOMMENDED IMPROVEMENTS:
    - Learn compression functions instead of prime frequencies
    - Implement dynamic chunking mechanism
    - Add multi-stage hierarchy
    - Consider byte-level input for better efficiency
    """
    def __init__(self, signal_length, m_tokens, token_vocab_size, token_embedding_dim, char_vocab_size, prime_tensor, device):
        super(End2EndModel, self).__init__()
        # NOTE: Model parameters - H-Net uses different configs per hierarchical stage
        self.signal_length = signal_length    # 512: Frequency resolution for signal encoding
        self.m_tokens = m_tokens              # 8: FIXED token count (H-Net varies per stage)
        self.token_embedding_dim = token_embedding_dim  # 64: Fixed representation size
        self.char_vocab_size = char_vocab_size  # Varies by dataset
        self.device = device
        
        # Register prime tensor for heuristic frequency encoding
        # NOTE: Could be replaced with learned embeddings per H-Net recommendations
        self.register_buffer("prime_tensor", prime_tensor)  # (vocab_size,) - prime frequencies
        
        # Model composition - single-stage vs. H-Net's multi-stage hierarchy
        self.pattern_encoder = PatternEncoder(signal_length, m_tokens, token_vocab_size, token_embedding_dim, dropout_prob=0.2, device=device)
        # NOTE: IntermediateTransformer is ADDITIONAL - not in H-Net paper
        self.intermediate_transformer = IntermediateTransformer(token_embedding_dim, num_layers=2, num_heads=4, dropout=0.1)
        self.pattern_decoder = PatternDecoder(m_tokens, token_embedding_dim, token_vocab_size, char_vocab_size)
    
    def forward(self, input_indices):
        # Main forward pass: Character indices → character logits
        # input_indices: (batch, context_length) - character-level input
        
        # Step 1: Signal encoding + tokenization → 8 fixed tokens
        # NOTE: Always produces 8 tokens regardless of input length (H-Net varies)
        token_flat, token_logits, token_embeds, normalized_signal = self.pattern_encoder(input_indices, self.prime_tensor)
        batch_size = token_flat.size(0)
        
        # Step 2: Additional transformer processing (NOT in H-Net)
        # NOTE: Processes 8 tokens through transformer - H-Net uses hierarchy
        refined_tokens = token_flat.view(batch_size, self.m_tokens, self.token_embedding_dim)  # (batch, 8, 64)
        transformer_out = self.intermediate_transformer(refined_tokens)  # (batch, 8, 64)
        
        # Step 3: Decode to character logits
        # NOTE: Single-stage decoding - H-Net uses multi-stage reconstruction
        transformer_flat = transformer_out.reshape(batch_size, -1)  # (batch, 512)
        char_logits = self.pattern_decoder(transformer_flat)  # (batch, char_vocab_size)
        
        # Return both final logits and intermediate token logits for analysis
        return char_logits, token_logits
    
    def forward_debug(self, input_indices):
        outputs = {}
        token_flat, token_logits, token_embeds, normalized_signal = self.pattern_encoder(input_indices, self.prime_tensor)
        outputs["context_signal_normalized"] = normalized_signal.detach().cpu().numpy()[0]
        outputs["token_embeds"] = token_embeds.detach().cpu().numpy()[0]
        outputs["token_logits"] = token_logits.detach().cpu().numpy()[0]
        outputs["token_flat"] = token_flat.detach().cpu().numpy()[0]

        batch_size = token_flat.size(0)
        refined_tokens = token_flat.view(batch_size, self.m_tokens, self.token_embedding_dim)
        transformer_out = self.intermediate_transformer(refined_tokens)
        transformer_flat = transformer_out.reshape(batch_size, -1)

        # Use PatternDecoder's decode_debug method on the transformer-enhanced tokens.
        dec_outputs = self.pattern_decoder.decode_debug(transformer_flat)
        outputs["pattern_logits"] = dec_outputs["pattern_logits"].detach().cpu().numpy()[0]
        outputs["refined_pattern"] = dec_outputs["refined_pattern"].detach().cpu().numpy()[0]
        outputs["pre_decoder_attn_weights"] = dec_outputs["pre_decoder_attn_weights"].detach().cpu().numpy()[0]
        outputs["char_logits"] = dec_outputs["char_logits"].detach().cpu().numpy()[0]
        
        return outputs

# =============================================================================
# Inference function: Generate text given a seed.
# =============================================================================
def generate_text(model, seed, idx_to_char, char_to_idx, seq_length, device, generation_length=100, temperature=0.3):
    # NOTE: Autoregressive generation - H-Net uses similar approach
    # Key difference: This works at character-level vs. H-Net's byte-level
    model.eval()
    generated = seed
    context = list(seed)
    
    for i in range(generation_length):
        # Prepare context window - fixed size sliding window
        # NOTE: Uses padding with spaces for short contexts - H-Net handles variable lengths
        if len(context) < seq_length - 1:
            context_window = [' '] * ((seq_length - 1) - len(context)) + context
        else:
            context_window = context[-(seq_length - 1):]  # Right-context window
        
        # Convert to indices and predict
        # NOTE: Character-level processing - H-Net uses byte-level for better efficiency
        input_indices = torch.tensor([[char_to_idx[ch] for ch in context_window]], dtype=torch.long, device=device)
        logits, _ = model(input_indices)  # Get next character logits
        
        # Temperature scaling for generation diversity
        scaled_logits = logits / temperature
        prob = F.softmax(scaled_logits, dim=-1)
        next_idx = torch.multinomial(prob, num_samples=1).item()
        next_char = idx_to_char[next_idx]
        
        # Update generation context
        generated += next_char
        context.append(next_char)
    
    return generated

# =============================================================================
# Chat mode: Interactively chat with the model.
# =============================================================================
def chat_mode(model, idx_to_char, char_to_idx, seq_length, device, generation_length=100):
    model.eval()
    print("Entering chat mode. Type 'quit' or 'exit' to stop.")
    while True:
        prompt = input("User: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        full_text = generate_text(model, prompt, idx_to_char, char_to_idx, seq_length, device, generation_length)
        # Extract the generated portion.
        response = full_text[len(prompt):].strip()
        print("Model:", response)

# =============================================================================
# Debugging: Plot intermediate layer outputs.
# =============================================================================
def plot_debug_outputs(outputs):
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    
    # Plot raw context signal (if available; here we only have normalized).
    axs[0,0].plot(outputs["context_signal_normalized"])
    axs[0,0].set_title("Context Signal (Normalized)")
    axs[0,0].set_xlabel("Signal Index")
    axs[0,0].set_ylabel("Amplitude")
    
    # Plot token embeddings.
    im1 = axs[1,0].imshow(outputs["token_embeds"], aspect='auto', cmap='viridis')
    axs[1,0].set_title("Token Embeddings")
    plt.colorbar(im1, ax=axs[1,0])
    
    # Plot token logits.
    im2 = axs[1,1].imshow(outputs["token_logits"], aspect='auto', cmap='viridis')
    axs[1,1].set_title("Token Logits")
    plt.colorbar(im2, ax=axs[1,1])
    
    # Plot flattened token representation.
    axs[2,0].plot(outputs["token_flat"])
    axs[2,0].set_title("Flattened Token Representation")
    axs[2,0].set_xlabel("Index")
    axs[2,0].set_ylabel("Value")
    
    # Plot pattern logits.
    axs[2,1].plot(outputs["pattern_logits"])
    axs[2,1].set_title("Pattern Logits")
    axs[2,1].set_xlabel("Pattern Index")
    axs[2,1].set_ylabel("Logit")
    
    # Plot refined pattern.
    axs[3,0].plot(outputs["refined_pattern"])
    axs[3,0].set_title("Refined Pattern")
    axs[3,0].set_xlabel("Pattern Index")
    axs[3,0].set_ylabel("Value")
    
    # Plot character logits.
    axs[3,1].plot(outputs["char_logits"])
    axs[3,1].set_title("Character Logits")
    axs[3,1].set_xlabel("Character Index")
    axs[3,1].set_ylabel("Logit")
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main training loop
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=64, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--seq_length", type=int, default=12, help="sequence length for training examples (context+target)")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--signal_length", type=int, default=512, help="number of points in the signal")
    parser.add_argument("--m_tokens", type=int, default=8, help="number of tokens to produce from the context signal")
    parser.add_argument("--token_vocab_size", type=int, default=256, help="vocabulary size for the tokenization layer")
    parser.add_argument("--token_embedding_dim", type=int, default=64, help="embedding dimension for tokens")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read the training text from presentation.txt.
    with open("presentation.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    # Build the character vocabulary.
    # NOTE: Character-level vocabulary - H-Net uses byte-level for better coverage
    vocab = sorted(set(text))  # Unique characters in training text
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}  # Character to index mapping
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}  # Index to character mapping
    char_vocab_size = len(vocab)
    print(f"Vocabulary size: {char_vocab_size}")
    
    # Generate prime numbers for heuristic frequency encoding
    # NOTE: This is a heuristic approach - H-Net learns compression functions instead
    primes = generate_primes(char_vocab_size)  # One prime per character
    prime_tensor = torch.tensor(primes, dtype=torch.float, device=device)  # (vocab_size,)
    
    # Build the dataset and dataloader.
    dataset = TextDataset(text, seq_length=args.seq_length, char_to_idx=char_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize the model.
    # NOTE: These hyperparameters are experimental and may not match H-Net optimal settings
    # H-Net paper uses different configurations for different stages
    model = End2EndModel(
        signal_length=args.signal_length,               # 512 - frequency resolution
        m_tokens=args.m_tokens,                         #   8 - FIXED token count (should be dynamic)
        token_vocab_size=args.token_vocab_size,         # 256 - soft tokenization vocab
        token_embedding_dim=args.token_embedding_dim,   #  64 - representation dimension
        char_vocab_size=char_vocab_size,
        prime_tensor=prime_tensor,                      # Heuristic frequency mapping
        device=device,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    # NOTE: Standard cross-entropy training - H-Net uses same approach
    # Key difference: Single loss vs. H-Net's hierarchical losses
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)  # (batch, seq_length-1)
            batch_targets = batch_targets.to(device)  # (batch,) - next character
            
            optimizer.zero_grad()
            logits, _ = model(batch_inputs)  # (batch, char_vocab_size)
            loss = criterion(logits, batch_targets)  # Cross-entropy on character prediction
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_inputs.size(0)  # Weighted by batch size
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {avg_loss:.4f}")
    
    # Inference after training.
    seed = "My name is "
    generated_text = generate_text(model, seed, idx_to_char, char_to_idx, args.seq_length, device, generation_length=200)
    print("Generated text:\n", generated_text)
    
    # Debug: Plot intermediate layer outputs.
    print("Plotting model intermediate outputs for the input: 'My name is '")
    debug_prompt = "My name is "
    debug_context = list(debug_prompt)
    if len(debug_context) < args.seq_length - 1:
        debug_context = [' '] * ((args.seq_length - 1) - len(debug_context)) + debug_context
    else:
        debug_context = debug_context[-(args.seq_length - 1):]
    debug_input = torch.tensor([[char_to_idx[ch] for ch in debug_context]], dtype=torch.long, device=device)
    debug_outputs = model.forward_debug(debug_input)
    plot_debug_outputs(debug_outputs)
    
    # Enter chat mode.
    chat_mode(model, idx_to_char, char_to_idx, args.seq_length, device, generation_length=200)

if __name__ == "__main__":
    main()
