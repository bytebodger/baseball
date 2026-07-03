"""Manual sanity check for PlayerEncoder: builds a random padded batch with
mixed sequence lengths (including an empty, no-history one), runs a forward
pass on CPU, then again on CUDA if available, and confirms the output shapes
match.

Not a pytest test (pytest only collects test_*.py) -- run directly:
    python tests/manual_shape_check.py
"""

import torch

from src.data.sequence_dataset import CONTINUOUS_FEATURES, MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB
from src.models.player_encoder import PlayerEncoder


def build_fake_batch(lengths: list[int], max_seq_len: int):
    """Random batch padded to the longest length in `lengths`, matching the
    shapes PlayerEncoder.forward expects."""
    batch_size = len(lengths)
    padded_len = max(max(lengths), 1)
    assert padded_len <= max_seq_len, "sequence length exceeds the encoder's positional embedding range"

    continuous = torch.zeros(batch_size, padded_len, len(CONTINUOUS_FEATURES))
    pitch_type = torch.zeros(batch_size, padded_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, padded_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, padded_len, dtype=torch.long)
    position = torch.zeros(batch_size, padded_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, padded_len, dtype=torch.bool)
    has_history = torch.zeros(batch_size, dtype=torch.bool)

    for i, length in enumerate(lengths):
        if length == 0:
            continue
        has_history[i] = True
        continuous[i, :length] = torch.randn(length, len(CONTINUOUS_FEATURES))
        pitch_type[i, :length] = torch.randint(0, len(PITCH_TYPE_VOCAB), (length,))
        outcome[i, :length] = torch.randint(0, len(OUTCOME_VOCAB), (length,))
        matchup[i, :length] = torch.randint(0, len(MATCHUP_VOCAB), (length,))
        position[i, :length] = torch.arange(length)
        padding_mask[i, :length] = False

    return continuous, pitch_type, outcome, matchup, position, padding_mask, has_history


def run_forward(model, batch, device):
    model = model.to(device)
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        output = model(*batch)
    return output


def main() -> None:
    torch.manual_seed(0)
    model = PlayerEncoder.from_yaml()
    model.eval()

    # Mix of lengths: some real history of varying size, a couple of
    # zero-length (no-history) sequences, and one right up near the cap.
    lengths = [50, 0, 12, model.config.max_seq_len - 1, 1, 0, 7]
    batch = build_fake_batch(lengths, model.config.max_seq_len)

    cpu_output = run_forward(model, batch, torch.device("cpu"))
    print(f"CPU output shape: {tuple(cpu_output.shape)}")

    if not torch.cuda.is_available():
        print("CUDA not available on this machine -- skipping the CUDA forward pass.")
        return

    cuda_output = run_forward(model, batch, torch.device("cuda"))
    print(f"CUDA output shape: {tuple(cuda_output.shape)}")

    assert cpu_output.shape == cuda_output.shape, "CPU and CUDA output shapes differ"
    print("OK: CPU and CUDA output shapes match, no errors thrown.")


if __name__ == "__main__":
    main()
