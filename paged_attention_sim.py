import numpy as np

class VirtualMemoryBlock:
    def __init__(self, block_size: int = 16, hidden_dim: int = 4096):
        self.block_size = block_size
        self.tokens = np.zeros((block_size, hidden_dim), dtype=np.float16)
        self.attention_scores = np.zeros(block_size, dtype=np.float32)
        self.active_count = 0

class PagedKVCache:
    """Simulates vLLM PagedAttention with gradient-based block eviction."""
    def __init__(self, total_vram_mb: int = 1024, block_size: int = 16):
        self.block_size = block_size
        self.max_blocks = (total_vram_mb * 1024 * 1024) // (block_size * 4096 * 2)
        self.block_table = [VirtualMemoryBlock(block_size) for _ in range(self.max_blocks)]
        
    def allocate_and_evaluate(self, token_embeddings: np.ndarray, attention_gradient: np.ndarray):
        """Allocates tokens to blocks. Evicts lowest-scoring block if OOM."""
        if self.max_blocks == 0:
            return
            
        # Mock eviction logic
        block_scores = [np.mean(b.attention_scores) for b in self.block_table]
        weakest_block_idx = np.argmin(block_scores)
        
        # Overwrite weakest block
        self.block_table[weakest_block_idx].tokens[:len(token_embeddings)] = token_embeddings
        self.block_table[weakest_block_idx].attention_scores[:len(attention_gradient)] = attention_gradient
        print(f"Evicted block {weakest_block_idx}, allocated new tokens.")

if __name__ == "__main__":
    cache = PagedKVCache(total_vram_mb=64)
    cache.allocate_and_evaluate(np.random.randn(16, 4096), np.random.randn(16))
