PST SeMoE Sequential Tuning

This directory contains sequential tuning scripts for the non-inherited SeMoE
hyperparameters used in the current class-aware SeMoE + baseline decoder setup.

Included scripts:
- `tune_loss_balance_weight.sh`
- `tune_class_embed_dim.sh`

Skipped on purpose because they are inherited from the original PEAFusion
training stack and you asked not to tune inherited parameters here:
- `SOLVER.BASE_LR`
- `SEED`
- `SOLVER.BACKBONE_MULTIPLIER`

Behavior:
- each script tunes exactly one parameter family
- each candidate runs for 400 epochs worth of steps
- the script writes a sweep summary into the corresponding checkpoint folder
- after one script finishes, keep the best run and move to the next script
