# RQ4 — Cross-Attention & Error Analysis
- Split: `train.npy`; Inference: **blend** (α=0.80); thresholds from VAL (`th_fβ=1.5`).
- F8: token→atom heatmaps + RDKit overlays for 1 TP and 1 FP (if available).
  - TP example: label=NR-ER, idx=6262
  - FP example: label=NR-ER, idx=2594
- T4: `T4_false_positives.csv`, `T4_false_negatives.csv`, `motifs_summary.csv` (simple SMARTS library).