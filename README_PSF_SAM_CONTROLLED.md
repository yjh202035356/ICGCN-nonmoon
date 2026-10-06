# Controlled PSF-SAM baseline

Uses:
- official PSF-SAM architecture code
- official freeze policy:
  polyp_feature / Domain_generalization / Adapter_mlp / linear_classic
- the same Kvasir train/val/test split and 1024 preprocessing as CRSS-SAM
- the same foreground Dice style for controlled comparison

Paper-reported settings preserved:
- ViT-H / 1024 input
- lr = 1e-4
- 30 epochs
- LR x0.1 every 10 epochs
- validation each epoch, best checkpoint for testing

The public paper/repo do not specify optimizer, batch size, or exact loss.
Controlled choices here:
- AdamW
- batch size 1, gradient accumulation 4
- CE + foreground soft-Dice

Do not call this an exact official reproduction.
Use the label:
PSF-SAM (official architecture, controlled reimplementation)
