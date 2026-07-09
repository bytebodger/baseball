"""Manual sanity check for LongHistoryEncoder: builds a random padded batch
across 8 players with wildly different amounts of game history (roughly 400,
250, 50, 15, 8, 3, 1, and 0 games -- from a long-tenured veteran down to a
career cold start), each game given a random pitch count (20-150, to
simulate real game-length variation), runs a forward pass on CPU, then again
on CUDA if available, and confirms the output shapes match.

Uses the real default configs (configs/player_encoder.yaml,
configs/career_encoder.yaml) rather than a small test config, since the
point is to sanity-check the production-sized model actually handles the
full 400-chunk x 150-pitch shape without erroring or blowing up memory --
expect this to take a while on CPU (a 4-layer, 4-head transformer runs over
8*400=3200 flattened per-game sequences at the chunk level, then again over
8 401-length sequences at the career level).

Not a pytest test (pytest only collects test_*.py) -- run directly:
    python tests/manual_shape_check_long_history.py
"""

import torch

from src.data.sequence_dataset import CONTINUOUS_FEATURES, MATCHUP_VOCAB, OUTCOME_VOCAB, PITCH_TYPE_VOCAB
from src.models.long_history_encoder import CareerEncoder, ChunkEncoder, LongHistoryEncoder

GAME_COUNTS = [400, 250, 50, 15, 8, 3, 1, 0]
MIN_PITCHES_PER_GAME = 20
MAX_PITCHES_PER_GAME = 150
MIN_DAYS_BETWEEN_GAMES = 1
MAX_DAYS_BETWEEN_GAMES = 6


def _days_before_cutoff_for_player(n_games: int) -> torch.Tensor:
    """Chronological order (oldest first, most recent last -- matching
    PlayerPitchSequenceDataset's own "most recent last" pitch convention):
    random 1-6 day gaps between consecutive starts, converted into "days
    before cutoff" counting back from the most recent game (last slot) = 0."""
    if n_games <= 1:
        return torch.zeros(n_games)
    gaps = torch.randint(MIN_DAYS_BETWEEN_GAMES, MAX_DAYS_BETWEEN_GAMES + 1, (n_games - 1,)).float()
    # gaps[i] = days between game i and game i+1. A game's days-before-cutoff
    # is the sum of every gap from that game up through the most recent one.
    days_before_cutoff = gaps.flip(0).cumsum(0).flip(0)
    return torch.cat([days_before_cutoff, torch.zeros(1)])


def build_fake_batch(game_counts: list[int], max_pitch_len: int):
    """Random nested batch: for each player, `game_counts[i]` real games
    (chunks) in chronological order, each with a random pitch count in
    [MIN_PITCHES_PER_GAME, MAX_PITCHES_PER_GAME], padded to
    max(game_counts) chunks and `max_pitch_len` pitches -- both the
    per-game pitch padding (chunk encoder input) and the per-player chunk
    padding (career encoder input) use the same "real data first, padding
    after" layout, masked via padding_mask exactly like a short pitch
    history is today.

    Returns the (chunk_pitch_sequences, days_before_cutoff,
    chunk_padding_mask, player_has_history) tuple LongHistoryEncoder.forward
    expects.
    """
    batch_size = len(game_counts)
    max_chunks = max(max(game_counts), 1)

    continuous = torch.zeros(batch_size, max_chunks, max_pitch_len, len(CONTINUOUS_FEATURES))
    pitch_type = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    outcome = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    matchup = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    position = torch.zeros(batch_size, max_chunks, max_pitch_len, dtype=torch.long)
    pitch_padding_mask = torch.ones(batch_size, max_chunks, max_pitch_len, dtype=torch.bool)
    chunk_has_history = torch.zeros(batch_size, max_chunks, dtype=torch.bool)

    days_before_cutoff = torch.zeros(batch_size, max_chunks)
    chunk_padding_mask = torch.ones(batch_size, max_chunks, dtype=torch.bool)
    player_has_history = torch.zeros(batch_size, dtype=torch.bool)

    for p, n_games in enumerate(game_counts):
        if n_games == 0:
            continue
        player_has_history[p] = True
        chunk_padding_mask[p, :n_games] = False
        days_before_cutoff[p, :n_games] = _days_before_cutoff_for_player(n_games)

        for c in range(n_games):
            n_pitches = int(torch.randint(MIN_PITCHES_PER_GAME, MAX_PITCHES_PER_GAME + 1, (1,)).item())
            chunk_has_history[p, c] = True
            continuous[p, c, :n_pitches] = torch.randn(n_pitches, len(CONTINUOUS_FEATURES))
            pitch_type[p, c, :n_pitches] = torch.randint(0, len(PITCH_TYPE_VOCAB), (n_pitches,))
            outcome[p, c, :n_pitches] = torch.randint(0, len(OUTCOME_VOCAB), (n_pitches,))
            matchup[p, c, :n_pitches] = torch.randint(0, len(MATCHUP_VOCAB), (n_pitches,))
            position[p, c, :n_pitches] = torch.arange(n_pitches)
            pitch_padding_mask[p, c, :n_pitches] = False

    chunk_pitch_sequences = {
        "continuous": continuous,
        "pitch_type": pitch_type,
        "outcome": outcome,
        "matchup": matchup,
        "position": position,
        "padding_mask": pitch_padding_mask,
        "has_history": chunk_has_history,
    }
    return chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, player_has_history


def run_forward(model, batch, device):
    model = model.to(device)
    chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, player_has_history = batch
    chunk_pitch_sequences = {k: v.to(device) for k, v in chunk_pitch_sequences.items()}
    days_before_cutoff = days_before_cutoff.to(device)
    chunk_padding_mask = chunk_padding_mask.to(device)
    player_has_history = player_has_history.to(device)
    with torch.no_grad():
        output = model(chunk_pitch_sequences, days_before_cutoff, chunk_padding_mask, player_has_history)
    return output


def main() -> None:
    torch.manual_seed(0)

    chunk_encoder = ChunkEncoder.from_yaml()
    career_encoder = CareerEncoder.from_yaml()
    model = LongHistoryEncoder(chunk_encoder, career_encoder)
    model.eval()

    print(f"Players: {len(GAME_COUNTS)}, game counts: {GAME_COUNTS}")
    print(f"Pitches per game: random in [{MIN_PITCHES_PER_GAME}, {MAX_PITCHES_PER_GAME}]")

    assert MAX_PITCHES_PER_GAME <= chunk_encoder.config.max_seq_len, (
        f"a game's pitch count (up to {MAX_PITCHES_PER_GAME}) exceeds ChunkEncoder's positional "
        f"embedding range ({chunk_encoder.config.max_seq_len})"
    )
    assert max(GAME_COUNTS) <= career_encoder.config.max_chunks, (
        f"a player's game count (up to {max(GAME_COUNTS)}) exceeds CareerEncoder's configured "
        f"max_chunks ({career_encoder.config.max_chunks}) -- not a hard architectural limit "
        "(ChunkTimeEncoding is continuous, not a lookup table), but this check keeps the example "
        "honest about what the cap is meant to cover"
    )

    batch = build_fake_batch(GAME_COUNTS, max_pitch_len=MAX_PITCHES_PER_GAME)

    print("Running forward pass on CPU (this may take a little while: "
          f"{len(GAME_COUNTS)}*{max(GAME_COUNTS)}={len(GAME_COUNTS) * max(GAME_COUNTS)} flattened chunks "
          "through the chunk encoder, then the career encoder)...")
    cpu_output = run_forward(model, batch, torch.device("cpu"))
    print(f"CPU output shape: {tuple(cpu_output.shape)}")
    assert not torch.isnan(cpu_output).any(), "CPU output contains NaN"

    if not torch.cuda.is_available():
        print("CUDA not available on this machine -- skipping the CUDA forward pass.")
        return

    print("Running forward pass on CUDA...")
    cuda_output = run_forward(model, batch, torch.device("cuda"))
    print(f"CUDA output shape: {tuple(cuda_output.shape)}")
    assert not torch.isnan(cuda_output).any(), "CUDA output contains NaN"

    assert cpu_output.shape == cuda_output.shape, "CPU and CUDA output shapes differ"
    print("OK: CPU and CUDA output shapes match, no errors thrown.")


if __name__ == "__main__":
    main()
