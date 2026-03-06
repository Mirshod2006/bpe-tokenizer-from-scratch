"""BPE merge operations and management."""

from typing import Dict, List, Tuple, Optional
from functools import lru_cache


class BPEMerger:
    """
    Handles BPE merge operations and rankings.
    Supports both custom trained merges and GPT-2 style ranked merges.
    """
    
    def __init__(self):
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges: Dict[Tuple[int, int], int] = {}
        
        # For GPT-2 style: {(string_A, string_B): rank} where lower rank = higher priority
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Reference to vocab (will be set by tokenizer)
        self.vocab: Dict[int, str] = {}
        self.inverse_vocab: Dict[str, int] = {}
    
    def set_vocab_references(self, vocab: Dict[int, str], inverse_vocab: Dict[str, int]):
        """Set references to the tokenizer's vocab dictionaries."""
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
    
    def add_merge(self, pair: Tuple[int, int], new_id: int):
        """Add a merge operation."""
        self.bpe_merges[pair] = new_id
    
    def add_ranked_merge(self, pair: Tuple[str, str], rank: int):
        """Add a ranked merge (GPT-2 style)."""
        self.bpe_ranks[pair] = rank
    
    def merge_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Merge tokens using custom trained merges.
        
        Args:
            token_ids: List of token IDs to merge
        
        Returns:
            List of merged token IDs
        """
        if not self.bpe_merges:
            return token_ids
        
        result = token_ids.copy()
        can_merge = True
        
        while can_merge and len(result) > 1:
            can_merge = False
            new_tokens = []
            i = 0
            
            while i < len(result) - 1:
                pair = (result[i], result[i + 1])
                if pair in self.bpe_merges:
                    new_tokens.append(self.bpe_merges[pair])
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(result[i])
                    i += 1
            
            if i < len(result):
                new_tokens.append(result[i])
            
            result = new_tokens
        
        return result
    
    def merge_with_ranks(self, symbols: List[str]) -> List[str]:
        """
        Merge symbols using GPT-2 style ranked merges.
        
        Args:
            symbols: List of symbol strings
        
        Returns:
            List of merged symbol strings
        """
        if not self.bpe_ranks:
            return symbols
        
        result = symbols.copy()
        
        while len(result) > 1:
            # Find all adjacent pairs
            pairs = set(zip(result, result[1:]))
            if not pairs:
                break
            
            # Find the pair with the lowest rank
            best_pair = None
            best_rank = float("inf")
            
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None or best_pair not in self.bpe_ranks:
                break
            
            # Merge all occurrences of the best pair
            first, second = best_pair
            new_symbols = []
            i = 0
            
            while i < len(result):
                if i < len(result) - 1 and result[i] == first and result[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(result[i])
                    i += 1
            
            result = new_symbols
        
        return result
    
    def get_merged_token_id(self, token: str) -> Optional[int]:
        """Get token ID for a merged token string."""
        return self.inverse_vocab.get(token)
    
    @lru_cache(maxsize=10000)
    def can_merge_pair(self, token_id1: int, token_id2: int) -> bool:
        """Check if a pair of token IDs can be merged."""
        return (token_id1, token_id2) in self.bpe_merges
    
    def to_dict(self) -> dict:
        """Convert merges to dictionary for serialization."""
        return {
            "bpe_merges": [{"pair": list(pair), "new_id": new_id} 
                          for pair, new_id in self.bpe_merges.items()],
            "bpe_ranks": [{"pair": list(pair), "rank": rank} 
                         for pair, rank in self.bpe_ranks.items()]
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create BPEMerger from dictionary."""
        merger = cls()
        
        for merge_data in data.get("bpe_merges", []):
            pair = tuple(merge_data["pair"])
            merger.bpe_merges[pair] = merge_data["new_id"]
        
        for rank_data in data.get("bpe_ranks", []):
            pair = tuple(rank_data["pair"])
            merger.bpe_ranks[pair] = rank_data["rank"]
        
        return merger