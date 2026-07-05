# Reward Function

## Design
- Per-bin penalty: `-0.5 * (avg_lab_deviation / 45)`
- Survival bonus: `+10 - 1 * LOS_norm`
- Death penalty: `-5`

## Rationale
See `src/config.py` for parameters.
