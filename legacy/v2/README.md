# Preliminary version (v2)

The first implemented design, discussed in the paper's Discussion ("Design rationale") and in the v2 vs. v3 figure.
It placed the normality memory, a learned uncertainty-gated fusion and a training-only planar committee *inside* the
segmentation network, and trained the fusion weights to agree with measured sequence-deletion effects.

On the same split it matched its baseline in mean Dice but lost 1.2 WT Dice points (p < 0.001) and degraded more
than the baseline when sequences were missing, which motivated the final design in `nora/`.
Results: `results/v2/`.

```bash
cd legacy/v2
python -m nora.train_loop --tag nora_v2 --data-root <BraTS2020_TrainingData> --cache cache --out runs --max-iters 15000 --batch 1
python -m nora.train_loop --tag baseline_v2 --no-ntpm --no-cmd --no-committee --no-egf --no-grade --data-root <...> --cache cache --out runs --max-iters 15000 --batch 1
```
