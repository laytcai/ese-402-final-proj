# Superconductor critical temperature: baseline

ESE 4020 final project, group FCC. We predict a superconductor's critical temperature (Tc, in kelvin) from 81 inputs computed from its chemical formula. The same material appears in the data many times, so we compare random 5-fold CV with 5-fold CV grouped by material.

Data: [UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data) (Hamidieh, 2018), CC BY 4.0. `baseline.py` downloads it on the first run.

## Run

Tested on Python 3.13. Takes about 3 minutes.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python baseline.py
.venv/bin/python plot.py
```

The seed (4020) and the tuning grids are at the top of `baseline.py`.

## Results

| File | Contents |
|---|---|
| `results/baseline.png` | RMSE per model under both schemes, and the gap between them |
| `results/summary.csv` | RMSE, chosen setting and 95% interval for each model and scheme |
| `results/paired.csv` | Paired differences between models and between schemes |
| `results/tuning.csv` | CV error for every lambda and k |
| `results/facts.json` | Data counts, the check against the paper, and the repeated-material floor |

The sealed 20% test set is never scored by this code.
