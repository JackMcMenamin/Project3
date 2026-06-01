"""
Word-embedding wrappers for the *word-level* transformer RSR experiments.

Both architectures share the same RSR contract:
  * Freeze the input embeddings + the first `num_frozen_layers` blocks;
    only the final block and a `hidden -> projection_dim` head are trainable.
  * Produce one projected vector per word by mean-pooling the relevant
    last-layer subword hidden states.

They differ only in tokenisation / pooling details, which the two subclasses
override. This is the logic that used to be duplicated verbatim between
`run_bert_seeds.py` and `run_gpt2_seeds.py`.

(These encode words in isolation. The contextual/sentence-level variant lives
in `experiments/contextual/` and reads cached vectors instead.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _WordEmbeddingBase(nn.Module):
    """Shared projection head + batch-embedding plumbing."""

    def _add_projection(self, hidden_size: int, projection_dim: int) -> None:
        """Attach the trainable head. Call after nn.Module.__init__ and after
        the backbone has been assigned (so self.hidden_size is known)."""
        self.hidden_size = hidden_size
        self.projection = nn.Linear(hidden_size, projection_dim)

    # --- subclasses must provide these ------------------------------------
    def _encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (last_hidden_state, attention_mask) for a batch of texts."""
        raise NotImplementedError

    def _prepare(self, word: str) -> str:
        """Map a bare word to the string actually fed to the tokenizer."""
        return word

    def _pool_single(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool one example's subword hidden states into one vector."""
        raise NotImplementedError

    # --- shared API -------------------------------------------------------
    def get_batch_embeddings(self, words: list[str], batch_size: int = 64) -> dict:
        embeddings: dict[str, torch.Tensor] = {}
        texts = [self._prepare(w) for w in words]
        grad_ctx = torch.enable_grad() if self.training else torch.no_grad()
        for i in range(0, len(words), batch_size):
            batch_words = words[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            with grad_ctx:
                hidden_states, attention_mask = self._encode(batch_texts)
            for j, word in enumerate(batch_words):
                vec = self._pool_single(hidden_states[j], attention_mask[j].bool())
                embeddings[word] = self.projection(vec)
        return embeddings


class BERTWordEmbeddings(_WordEmbeddingBase):
    """BERT (uncased). CLS/SEP stripped before pooling subword tokens."""

    def __init__(self, model_name="bert-base-uncased", projection_dim=128, num_frozen_layers=11):
        nn.Module.__init__(self)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self._add_projection(self.bert.config.hidden_size, projection_dim)

        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for i in range(num_frozen_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = False
        for p in self.bert.pooler.parameters():
            p.requires_grad = False

        self.to(DEVICE)

    def _encode(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        out = self.bert(**tokens)
        return out.last_hidden_state, tokens["attention_mask"]

    def _pool_single(self, hidden, mask):
        word_hidden = hidden[mask, :]
        if word_hidden.shape[0] > 2:  # drop [CLS] and [SEP]
            word_hidden = word_hidden[1:-1, :]
        return word_hidden.mean(dim=0)


class GPT2WordEmbeddings(_WordEmbeddingBase):
    """GPT-2. A leading space is prepended so the word tokenises as mid-sentence."""

    def __init__(self, model_name="gpt2", projection_dim=128, num_frozen_layers=11):
        nn.Module.__init__(self)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 has no pad token
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self._add_projection(self.gpt2.config.hidden_size, projection_dim)

        for p in self.gpt2.wte.parameters():
            p.requires_grad = False
        for p in self.gpt2.wpe.parameters():
            p.requires_grad = False
        for i in range(num_frozen_layers):
            for p in self.gpt2.h[i].parameters():
                p.requires_grad = False
        for p in self.gpt2.ln_f.parameters():
            p.requires_grad = False

        self.to(DEVICE)

    def _prepare(self, word):
        return " " + word

    def _encode(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        out = self.gpt2(**tokens)
        return out.last_hidden_state, tokens["attention_mask"]

    def _pool_single(self, hidden, mask):
        # GPT-2 has no special tokens to strip; pool all real positions.
        return hidden[mask, :].mean(dim=0)


# Registry so a runner can pick a wrapper by name.
WRAPPERS = {
    "bert": BERTWordEmbeddings,
    "gpt2": GPT2WordEmbeddings,
}
