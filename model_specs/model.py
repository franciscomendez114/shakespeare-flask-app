import torch
import torch.nn as nn
import sys
import time
from collections import Counter
from typing import List, Tuple
import json
import os
import re
from tqdm import tqdm


# Custom-made Tokenizer Class

# Initialize Custom BPE tokenizer
class Tokenizer:
    def __init__(self, text: str, special_tokens: List[str]):
        self.text = text
        self.special_tokens = special_tokens
        self.vocab = {}
        self.pre_tokens = self.pre_tokenize_str(text)

        # Store tokens as list of symbols (initially chars)
        self.token_sequences = {token: list(token) for token in self.pre_tokens}
        self.vocab = {char: i for i, char in enumerate(sorted(set(char for token in self.token_sequences.values() for char in token)))}

        self.index_to_token = {i: t for t, i in self.vocab.items()}
        self.token_to_index = self.vocab.copy()

        # Merges performed in order
        self.merges = []
        self.pair_counts = Counter()
        self.update_pair_counts()

    def pre_tokenize_str(self, string: str) -> List[str]:
        return re.findall(r"""\n|\s?\w+|[?,;:!'".\-\$&]""", string, re.IGNORECASE)

    def update_pair_counts(self):
        self.pair_counts.clear()
        for seq in self.token_sequences.values():
            for a, b in zip(seq, seq[1:]):
                self.pair_counts[(a, b)] += 1

    def get_most_common_pair(self) -> Tuple[str, str]:
        return self.pair_counts.most_common(1)[0][0]

    def merge_pair(self, pair: Tuple[str, str]):
        a, b = pair
        ab = a + b
        self.merges.append((a, b))
        self.vocab[ab] = len(self.vocab)
        self.token_to_index[ab] = self.vocab[ab]
        self.index_to_token[self.vocab[ab]] = ab

        new_token_sequences = {}
        for token, seq in self.token_sequences.items():
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(ab)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_token_sequences[token] = new_seq

        self.token_sequences = new_token_sequences
        self.update_pair_counts()

    def train(self, vocab_size: int):
        for _ in tqdm(range(vocab_size - len(self.vocab))):
            if not self.pair_counts:
                break
            most_common_pair = self.get_most_common_pair()
            self.merge_pair(most_common_pair)

    def apply_merges(self, token: str) -> List[str]:
        seq = list(token)
        for a, b in self.merges:
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    seq[i:i+2] = [a + b]
                else:
                    i += 1
        return seq

    def encode(self, text: str) -> List[int]:
        tokens = self.pre_tokenize_str(text)
        output_ids = []
        for token in tokens:
            symbols = self.apply_merges(token)
            output_ids.extend(self.token_to_index[sym] for sym in symbols)
        return output_ids

    def decode(self, token_ids: List[int]) -> str:
        symbols = [self.index_to_token[token_id] for token_id in token_ids]
        return ''.join(symbols)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Tokenizer':
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild the tokenizer object
        tokenizer = cls(text="", special_tokens=data["special_tokens"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data["merges"]
        tokenizer.token_to_index = data["vocab"]
        tokenizer.index_to_token = {i: t for t, i in data["vocab"].items()}
        return tokenizer

# LLM Encoder 
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_size, embed_dim)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.token_embed(x) + self.pos_embed(pos)

# LLM Transformer Block 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, context_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # Register a fixed causal mask buffer (T x T)
        mask = torch.tril(torch.ones(context_size, context_size)).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.size()

        # Apply causal mask (True=keep, False=mask out)
        causal_mask = getattr(self, 'causal_mask')[:T, :T]
        attn_mask = ~causal_mask.bool()  # Flip to True=mask out

        # Self-attention
        attn_output, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)

        # Feedforward
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.ln2(x)
        return x

# LLM Transformer Model

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout, context_size):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, context_size)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, context_size)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

tokenizer = Tokenizer.load(os.path.join(os.path.dirname(__file__), 'customTokenizer.json'))

vocab_size = len(tokenizer.vocab)
embed_dim = 256
num_heads = 4
num_layers = 2
dropout = 0.2
context_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(vocab_size, embed_dim, num_heads, num_layers, dropout, context_size).to(device)

# Load Model Weights

state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'model.pth'), map_location=torch.device('cpu'))
# If keys are prefixed with '_orig_mod.', strip it
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    state_dict = new_state_dict
model.load_state_dict(state_dict)
model.eval()

def generate_text(prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 200):
    """
    Generate text using the loaded model.
    
    Args:
        prompt: The initial text to start generation from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: Number of top tokens to consider for sampling
    
    Returns:
        Tuple of (generated text as a string, list of generated token IDs)
    """
    model.eval()
    
    # Encode the prompt
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens with rolling context window
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Always use only the last context_size tokens for the model
            input_context = context[:, -context_size:]
            logits = model(input_context)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            context = torch.cat([context, next_token], dim=1)
    
    # Decode the generated tokens
    generated_tokens = context[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text, generated_tokens

def get_tokenization_info(text: str):
    """
    Get detailed tokenization information for visualization.
    
    Args:
        text: The text to tokenize
    
    Returns:
        Dictionary containing:
        - tokenized_text: The original text with token boundaries marked
        - tokens: List of individual tokens
        - token_positions: List of (start, end) positions for each token in the original text
    """
    # Encode the text to get token IDs
    token_ids = tokenizer.encode(text)
    
    # Get individual tokens by decoding each token ID
    tokens = []
    token_positions = []
    
    current_pos = 0
    for token_id in token_ids:
        token = tokenizer.index_to_token[token_id]
        tokens.append(token)
        start_pos = current_pos
        end_pos = current_pos + len(token)
        token_positions.append((start_pos, end_pos))
        current_pos = end_pos
    
    return {
        'tokenized_text': text,
        'tokens': tokens,
        'token_positions': token_positions,
        'token_ids': token_ids
    }




