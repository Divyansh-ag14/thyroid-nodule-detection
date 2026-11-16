# Improvement candidate: FibonacciNet with BatchNorm in the pcb branches

Written before any `fibonacci_net_pcbnorm` result existed.

**Motivation.** Phase 1 showed FibonacciNet non-inferior to the standard CNN on test, but consistently lower on
validation (seeds 0.906 / 0.903 / 0.899 vs CNN 0.914 / 0.917 / 0.916) and with a lower training AUROC (0.93 vs 0.945),
pointing to an optimisation limit rather than overfitting. The pcb branches concatenate unnormalized Avg-2Max output
(`avg - 2*max`, large and mostly negative) with BN+ReLU main-path features. The candidate adds one BatchNorm at the end of
each pcb branch (+192 parameters, same FLOPs).

**Decision rule (validation AUROC only; the test set is not consulted):**

1. Train seed 0 with the shared recipe (cosine, peak LR 2e-3, standard augmentation, 40 epochs).
2. Continue only if its best val AUROC exceeds 0.906 (best original seed) by at least 0.005.
3. If continued, train seeds 1 and 2. Adopt only if the 3-seed mean val AUROC exceeds 0.903 (original mean) by at
   least 0.005.
4. If adopted, the variant is benchmarked against the standard CNN with the identical protocol and reported alongside
   the original FibonacciNet, whose results are kept.

## Outcome

Seed 0 reached a best val AUROC of **0.901**, below the 0.911 required by step 2 (0.906 + 0.005). Per the rule, seeds
1 and 2 were not trained and the variant was **not adopted**; the original FibonacciNet remains the reported model.
Training AUROC rose (0.948 at the last epoch, vs 0.93 for the original) without a validation gain, i.e. the
normalization made the training set easier to fit but did not improve generalisation. The run's history and
predictions are archived in `reports/runs/fibonacci_net_pcbnorm/seed0/`.
