import torch
from torch import nn


class RNNAutocompletion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=pad_token_id,
        )
        self.embeddings_norm = nn.LayerNorm(dim)
        self.rnn = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(in_features=dim, out_features=vocab_size, bias=False)
        self.linear.weight = self.embeddings.weight

    def forward(self, input_ids):

        x = self.embeddings(input_ids)
        x = self.embeddings_norm(x)

        out, (h, c) = self.rnn(x)
        out = self.norm(out)

        logits = self.linear(out)

        return logits, (h, c)

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        device = start_tokens.device
        B, K = start_tokens.shape

        x = self.embeddings(start_tokens)
        x = self.embeddings_norm(x)
        _, (h, c) = self.rnn(x)

        lengths = (start_tokens != self.pad_token_id).sum(dim=1).clamp(min=1)
        batch_idx = torch.arange(B, device=device)
        prev = start_tokens[batch_idx, lengths - 1]

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        out_tokens = []

        for _ in range(max_new_tokens):
            emb = self.embeddings(prev).unsqueeze(1)
            emb = self.embeddings_norm(emb)
            o, (h, c) = self.rnn(emb, (h, c))
            o = self.norm(o)
            logits = self.linear(o.squeeze(1))

            next_id = torch.argmax(logits, dim=-1)
            out_tokens.append(next_id.unsqueeze(1))

            finished |= next_id.eq(self.eos_token_id)
            prev = torch.where(
                finished, torch.full_like(next_id, self.eos_token_id), next_id
            )

            if finished.all():
                break

        if not out_tokens:
            return torch.empty(B, 0, dtype=start_tokens.dtype, device=device)
        return torch.cat(out_tokens, dim=1)
