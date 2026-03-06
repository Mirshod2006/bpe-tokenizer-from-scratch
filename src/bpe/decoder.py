"""BPE decoding module."""

from typing import List

from .constants import GPT2_SPACE_TOKEN, NEWLINE_TOKEN


class BPEDecoder:
    """
    Handles decoding token IDs back to text.
    """
    
    def __init__(self, vocab: dict[int, str]):
        """
        Initialize decoder with vocabulary.
        
        Args:
            vocab: Token ID to token string mapping
        """
        self.vocab = vocab
        self.id_to_token = vocab  # Alias for clarity
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        if not token_ids:
            return ""
        
        decoded_parts = []
        
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
            
            token = self.vocab[token_id]
            decoded_parts.append(self._format_token(token, i == 0))
        
        return self._join_decoded_parts(decoded_parts)
    
    def _format_token(self, token: str, is_first: bool) -> str:
        """
        Format a single token for output.
        
        Args:
            token: Token string
            is_first: Whether this is the first token
        
        Returns:
            Formatted token string
        """
        if token == NEWLINE_TOKEN:
            return token
        elif token.startswith(GPT2_SPACE_TOKEN):
            return " " + token[1:]
        else:
            return token
    
    def _join_decoded_parts(self, parts: List[str]) -> str:
        """
        Join decoded parts with proper spacing.
        
        Args:
            parts: List of formatted token strings
        
        Returns:
            Properly joined text
        """
        result = ""
        for i, part in enumerate(parts):
            if i == 0:
                result = part
            elif part == NEWLINE_TOKEN:
                # Ensure space before newline if needed
                if result and not result.endswith(" "):
                    result += " "
                result += part
            else:
                result += part
        
        return result
    
    def decode_with_metadata(self, token_ids: List[int]) -> dict:
        """
        Decode and return metadata about the decoding process.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Dictionary with decoded text and token metadata
        """
        tokens = []
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocabulary")
            
            tokens.append({
                "id": token_id,
                "token": self.vocab[token_id],
                "type": self._get_token_type(self.vocab[token_id])
            })
        
        return {
            "text": self.decode(token_ids),
            "num_tokens": len(token_ids),
            "tokens": tokens
        }
    
    def _get_token_type(self, token: str) -> str:
        """Categorize token type."""
        if token == NEWLINE_TOKEN:
            return "newline"
        elif token.startswith(GPT2_SPACE_TOKEN):
            return "word_with_space"
        elif token.startswith("<|") and token.endswith("|>"):
            return "special"
        elif len(token) == 1:
            return "character"
        else:
            return "subword"