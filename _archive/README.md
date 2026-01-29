# Archive

Archived on 2026-01-29.

Old evaluation scripts replaced by the unified experiment pipeline in `experiments/`.

## Archived Files

- `evaluate_rule_model.py` - V1 baseline
- `evaluate_rule_model_v2.py` - V2 with prospective features
- `evaluate_rule_model_v3.py` - V3 with feature engineering (65% accuracy)
- `evaluate_rule_model_v3_optimized.py` - Grid search optimization

## Archived Results

- `results/` - V1 results
- `results_v2/` - V2 results
- `results_v3/` - V3 results
- `results_v3_optimized/` - Optimized weights

## New Pipeline

Use the new experiment pipeline instead:

```bash
# Run an experiment
python -m experiments.run experiments/configs/v3_features.yaml

# List past experiments
python -m experiments.run --list
```

The new pipeline provides:
- YAML-driven configuration
- Automatic experiment logging
- Pass/fail tracking (artifacts only saved for passing experiments)
- Easy comparison across experiments
