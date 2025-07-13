# H-Net
- hnet paper https://arxiv.org/html/2507.07955v1

<table width="100%">
  <tr><td><img src="https://github.com/goombalab/hnet/raw/main/assets/code.gif" alt="Code" width="100%"></td></tr>
  <tr><td><img src="https://github.com/goombalab/hnet/raw/main/assets/chinese.gif" alt="Chinese" width="100%"></td></tr>
</table>

## Theoretical Highlights (from the H-Net Paper)

1.  **End-to-End Learning at the Byte Level:** This is the most groundbreaking point. H-Net completely eliminates the pre-processing step of tokenization with a fixed vocabulary (like BPE or WordPiece). Instead, it learns directly from raw bytes, allowing the model to build its own representations.

2.  **Dynamic Chunking (DC):** Instead of fixed-length or predetermined tokens, H-Net dynamically learns to group bytes into meaningful "chunks." This process is context-dependent, meaning the same sequence of bytes can be chunked differently depending on what precedes and follows it.

3.  **Hierarchical Architecture:** The model processes information at multiple levels of abstraction. The lowest level operates on bytes, creating chunks. Higher levels operate on the representations of these chunks, enabling the model to learn complex structures and long-range dependencies.

4.  **Superior Performance and Scalability:** The paper claims that H-Net not only matches but surpasses token-based Transformer models of the same scale, especially on large datasets. The hierarchical architecture shows better scalability.

5.  **Robustness:** Being byte-level, the model is very robust to spelling errors, out-of-vocabulary words, and other text variations that often pose challenges for fixed-vocabulary tokenizers.

## Source Code Highlights (Implementation)

1.  **Implementation of Dynamic Chunking (`dc.py`):**
    *   `RoutingModule`: This is the "brain" of the chunking mechanism. It calculates the probability of a token being a "boundary" of a chunk by comparing the cosine similarity between adjacent token representations.
    *   `ChunkLayer` & `DeChunkLayer`: These layers handle the gathering of tokens into chunks based on the boundaries identified by `RoutingModule` and then "un-chunk" them back to the original sequence after processing at a higher level.

2.  **Recursive Architecture (`hnet.py`):** The `HNet` class recursively calls itself to build the layers of the hierarchy. The `if self.is_innermost:` structure is a clear demonstration of this design.

3.  **Hybrid Architecture (`block.py`, `isotropic.py`):** The source code allows for flexible selection between Mamba blocks (`'m'`, `'M'`) and Attention blocks (`'t'`, `'T'`) within the same architecture. This allows for combining the strengths of both: Mamba's efficient processing of long sequences and Attention's ability to capture complex relationships.

4.  **FlashAttention Integration (`mha.py`):** The use of FlashAttention shows that the source code is optimized for high performance on modern GPUs.

5.  **Clever Kernel Reuse (`dc.py`):** The `DeChunkLayer` reuses the `mamba_chunk_scan_combined` kernel from Mamba2 to efficiently perform the EMA (Exponential Moving Average) operation, demonstrating a deep understanding of the underlying libraries.

6.  **Complete Language Model (`mixer_seq.py`):** The `HNetForCausalLM` class encapsulates the entire logic, from input embeddings to the final lm_head, and inherits from `GenerationMixin` to provide convenient text generation methods (`generate`, `step`).
