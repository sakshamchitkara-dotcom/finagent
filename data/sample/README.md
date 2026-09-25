# SYNTHETIC sample data

Every CSV in this folder is **synthetic** (not real market prices), generated
deterministically by `scripts/gen_sample_data.py` (seed 42, GBM with regime
switching). Tickers are prefixed `SYN_` so they can't be mistaken for real
securities. Regenerate with:

    python scripts/gen_sample_data.py
