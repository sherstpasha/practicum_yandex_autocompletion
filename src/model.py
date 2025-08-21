import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class RNNAutocompletion(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
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

    def forward(self, input_ids, lengths):

        out = self.embeddings(input_ids)
        out = self.embeddings_norm(out) 
        out, _ = self.rnn(out)
        out = self.norm(out)

        batch_idx = torch.arange(out.size(0), device=out.device)
        last_h = out[batch_idx, lengths - 1, :]
        
        out = self.linear(last_h)
        return out

    @torch.no_grad()
    def generate(self, start_tokens: torch.Tensor, max_tokens: int):

        device = start_tokens.device

        generated = start_tokens.clone().to(device)
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=device)
        new_tokens = []

        for _ in range(max_tokens):
            lengths = (
                (generated != self.pad_token_id).sum(dim=1).to(torch.long).clamp_min(1)
            )

            logits = self(generated, lengths)

            next_id = torch.argmax(logits, dim=-1)
            next_id = torch.where(
                finished, torch.full_like(next_id, self.pad_token_id), next_id
            )

            new_tokens.append(next_id.unsqueeze(1))
            generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)

            finished |= next_id.eq(self.pad_token_id)
            if finished.all():
                break

        return torch.cat(new_tokens, dim=1)
