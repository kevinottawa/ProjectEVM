# EVM Simulator Status

The original C++ simulator merged equal expert IDs across different layers. Its superseded outputs are excluded from this repository and must not be used as paper evidence.

Canonical reproduction:

```powershell
python scripts/train_layer_aware_predictor.py
python scripts/plot_production_evm.py
```

Canonical outputs:

- `docs/tables/layer_aware_predictor_results.csv`
- `results/learned_predictor/router_probability_predictor.npz`
- `docs/figures/production_evm/layer_aware_predictor_hit_rate.png`
