# Experiments Log (Deep Fund L1)

This document records run configurations, offline diagnostics (especially seeds-only SSE/MSE), ensemble weights, calibration effects, and leaderboard outcomes. It is intended to guide further data-driven improvements without hardcoding final weights.

## Conventions
- Seeds-only SSE/MSE: computed on pairs where both repos are seeds (matches leaderboard slice).
- Ensemble weights: simplex-optimized on seeds-only pairs unless noted.
- Temperature: global post-fit scale t fitted on seeds-only pairs.
- Category calibration: post-fit per-category offsets; often disabled in best runs.

## Dataset/Environment
- Train: `dataset/train.csv`
- Seeds/Test: `seedRepos.json`, `dataset/test.csv`
- Script: `script.py`
- Python 3.9+, NumPy
- External data: public GitHub API (use `GITHUB_TOKEN` to avoid throttling). Cached to `external_metrics.json`.

## Summary Table

| ID | Command (PowerShell) | Key Flags | Seeds-SSE | Seeds-MSE | Ensemble | Temp t | Cat Calib | Leaderboard | Notes |
|---|---|---|---:|---:|---|---:|---|---:|---|
| A1 | `python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --lam_cat 0.20 --seeds_boost 1.6` | cat calib on | 563.1608 | 3.352148 | 100% seeds_only | 1.0713 | ON | n/s | Slightly worse after cat-calib pre-temp improved but post-temp hurt |
| A2 | `python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --seeds_boost 1.6 --no_calibrate_cats` | no cat calib | 562.1983 | 3.346419 | 100% seeds_only | 1.0744 | OFF | n/s | Strong offline; basis for best leaderboard |
| A3 | `python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --seeds_boost 1.7 --no_calibrate_cats` | no cat calib | 562.1497 | 3.346129 | 100% seeds_only | 1.0738 | OFF | 5.0295 | Best leaderboard so far |
| B1 | `python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --seeds_boost 1.7 --no_calibrate_cats --cap_logmult 6.5` | mild cap | 556.8861 | 3.314798 | 100% seeds_only | 1.0107 | OFF | 5.2487 | Lower offline SSE but worse leaderboard (proxy mismatch) |
| C1 | `python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --juror_scale --lam_t 0.3 --t_clip_lo 0.8 --t_clip_hi 1.25 --huber --huber_delta 1.0 --lam_cat 0.18 --seeds_boost 1.1 --cap_logmult 5.5` | juror-scale + Huber | 580.7652 | 3.456936 | ~81% seeds_only | 1.0213 | ON | n/s | Robust fit helped structure but not SOTA here |
| D0 | Older baselines (pre-refinements) | — | — | — | — | — | — | 5.7865 / 5.2267 | Earlier attempts before leaderboard alignment |
| E1 | `python script.py --use_features --fetch_metrics --kfold 8 --tune_seeds_boost --tune_robust --no_calibrate_cats --verbose --outdir .` | tuned seeds_boost & robust; no cat calib | 588.8244 | 3.504907 | seeds_only 0.908, feature 0.092 | 1.0247 | OFF | 4.8827 | CV: lam_z=0.8, lam_theta=3.0, p=0.0, seeds_boost=1.4, huber delta=0.8; juror_scale=False |
| E2 | `python script.py --use_features --kfold 5 --verbose --tune_seeds_boost --tune_robust --tune_calibration --outdir .` | CV calib OFF (both); ensemble on seeds-only | 594.5886 | 3.539218 | feat 0.147, seeds 0.790, prior 0.063 | OFF | OFF | 4.8916 | Full data; Huber delta=0.8; cats/temp disabled by CV |
| E3 | `python script.py --rng_seed 42 --kfold 5 --verbose --exclude_nonseeds --no_ensemble --tune_seeds_boost --tune_robust` | seeds-only training; single model | 594.6268 | 3.539445 | — | 1.0162 | ON | 4.9279 | Category calib + temp applied; worse LB |
| E4 | `python script.py --use_features --kfold 5 --verbose --tune_seeds_boost --tune_robust --tune_balance_by_juror --juror_bias --reliability` | juror bias + reliability | 612.1536 | 3.643772 | feat 0.240, seeds 0.686, prior 0.073 | 1.0363 | ON | n/s | Offline SSE worsened; not submitted |
| F1 | `python script.py --use_features --kfold 8 --tune_seeds_boost --tune_robust --verbose --outdir .` | OSO boosts + cat calib ON | 1627.5445 | 4.351723 | seeds_only 1.000 | 1.0767 | ON | 4.5042 | OSO dep rank + funding + dev boosts working; BUT category calib made SPECS worse (26% vs target 8%) |

| G1 | `python script.py --use_features --kfold 5 --no_calibrate_cats --laplacian_weight 0.3 --laplacian_cat_weight 0.05 --private_pairs dataset/private_leaderboard.csv --proxy_weight 0.5 --tune_robust --lam_t_grid 0.3,0.5 --huber_delta_grid 3.0,4.0 --tune_seeds_boost --seeds_boost_grid 1.2,1.6 --cap_logmult_grid 0,4.5 --per_repo_cap 0.12 --shrink_eta 0.1` | features + Laplacians + caps/shrink + proxy 0.5 | — | — | — | — | OFF | 6.0030 | Regression; over-smoothing + caps hurt LB |
| G2 | `python script.py --use_features --kfold 5 --no_calibrate_cats --laplacian_weight 0.3 --laplacian_cat_weight 0.05 --private_pairs dataset/private_leaderboard.csv --proxy_weight 1.5 --tune_robust --lam_t_grid 0.3,0.5 --huber_delta_grid 3.0,4.0 --tune_seeds_boost --seeds_boost_grid 1.2,1.6 --cap_logmult_grid 0,4.5 --per_repo_cap 0 --shrink_eta 0` | features + stronger proxy + no caps/shrink | — | — | seeds_only 1.000 | 1.2366 | OFF | 4.5225 | Ensemble collapsed to seeds-only robust; improved LB |
| G3 | `python script.py --use_features --kfold 5 --no_calibrate_cats --laplacian_weight 0.15 --laplacian_cat_weight 0.0 --private_pairs dataset/private_leaderboard.csv --proxy_weight 2.0 --tune_robust --lam_t_grid 0.3,0.5 --huber_delta_grid 3.0,4.0 --tune_seeds_boost --seeds_boost_grid 1.2,1.6 --cap_logmult_grid 0,4.5 --per_repo_cap 0 --shrink_eta 0` | features + weaker Laplacian, no category Laplacian | — | — | seeds_only 1.000 | — | OFF | 4.5420 | Slight regression vs G2 |
| G4 | `python script.py --kfold 5 --no_calibrate_cats --no_ensemble --laplacian_weight 0.15 --laplacian_cat_weight 0.0 --private_pairs dataset/private_leaderboard.csv --proxy_weight 2.0 --tune_robust --lam_t_grid 0.3,0.5 --huber_delta_grid 3.0,4.0 --tune_seeds_boost --seeds_boost_grid 1.2,1.6 --cap_logmult_grid 0 --per_repo_cap 0 --shrink_eta 0` | non-features, seeds-only robust, proxy 2.0 | — | — | — | — | OFF | 4.5207 | Comparable to G2; large CV felt noisy due to many combos |
| G5 | `python script.py --use_features --kfold 3 --no_calibrate_cats --laplacian_weight 0.15 --laplacian_cat_weight 0.0 --private_pairs dataset/private_leaderboard.csv --proxy_weight 2.0 --tune_robust --lam_t_grid 0.5 --huber_delta_grid 3.0 --tune_seeds_boost --seeds_boost_grid 1.6 --cap_logmult_grid 0 --per_repo_cap 0 --shrink_eta 0 --no_ensemble` | features CV-lite (3-fold), no category Laplacian | — | — | — | — | OFF | 4.5070 | Best recent; seeds-only robust path |
| G6 | `python script.py [CV-lite repeat; minor variance]` | same family as G5 | — | — | — | — | OFF | 4.5124 | Within noise of G5 |
| **H1** | `python phase2_mixed_effects.py --cv --verbose --outdir .` | **Mixed-Effects Model with juror category biases** | **1627.54** | **4.3481** (base) / **2.2060** (mixed) | — | — | OFF | **4.4753** | **Phase 2**: λ_base=3.0, λ_beta=0.5; 49% training improvement; 0.64% LB improvement (minimal transfer to new jurors) |
| **H2** | `python phase3_advanced.py --outdir .` | **Phase 3: Minimal Tweaks** | — | — | base 0.90, dev 0.05, decent 0.05 | — | OFF | **4.4460** | **NEW BEST**: λ=0.8; minimal foundational boosts (1.15x-1.25x); slight category rebalance (TOOLS +10%, LANG +25%); 0.6% improvement from H1 |
| **H3** | `python phase5_constrained.py --verbose --outdir .` | **Phase 5: Constrained Juror-Aware Model** | — | **2.2855** (mixed) | — | — | OFF | **5.2195** | **FAILED**: λ_base=2.5, λ_beta=0.5; geth capped 23.96%→17.5%; Weight constraints backfired on current LB; juror modeling not effective for public test set |

n/s = not submitted

## Detailed Notes

- Seeds-only emphasis (`--seeds_boost` ≈ 1.6–1.7) consistently improved seeds-only SSE and often leaderboard; beyond ~1.7 shows diminishing returns.
- Ensemble optimization on seeds-only pairs typically collapses to the seeds-only candidate, which aligns with leaderboard scoring.
- Temperature calibration reduces seeds-only SSE by ~0.5–2.0; category calibration often over-corrects post-temperature and can hurt leaderboard.
- Juror balancing + tight caps (~4.2) degraded seeds-only SSE and leaderboard; robust methods (Huber) are preferable to hard caps.
- Adding richer features (languages/topics/releases, etc.) sometimes improves offline SSE (CV increases `lam_theta`), but leaderboard impacts vary; the seeds-only SSE proxy is strong but not perfect.
- Introducing per-juror additive bias and reliability down-weighting (E4) increased seeds-only SSE and did not improve leaderboard in near-term tests; keep OFF for leaderboard runs unless CV shows gains.

## Reproduce Current Best (4.5042) [NEW - Oct 2, 2025]

```powershell
python script.py --use_features --kfold 8 --tune_seeds_boost --tune_robust --verbose --outdir .
```

Expected logs:
- **OSO Loading**: `[OSO] Loaded 363 project entries`, `Coverage: 33/45 seeds`
- **OSO Boosts**: ethers.js 7.01x, hardhat 4.65x, alloy 4.48x, OZ 4.20x, geth 3.13x
- CV-picked `lam_z = 0.8`, `lam_theta = 3.0`, `p = 0.0`, `seeds_boost = 1.4`, `huber delta = 3.0`, `juror_scale = True, lam_t = 0.300`
- `[candidates]` Seeds-only diagnostics:
  - `feature_ridge_js_huber`: Seeds-SSE = 1650.225, Seeds-MSE = 4.412367
  - `seeds_only_ridge_js_huber`: Seeds-SSE = 1639.195, Seeds-MSE = 4.382875
  - `feature_prior_only`: Seeds-SSE = 2349.927, Seeds-MSE = 6.283229
- `[ensemble]` weights: `seeds_only_ridge_js_huber: 1.000` (100%)
- `[calib]` category offsets: `SPECS:+0.3103` (problematic - increased overweight)
- `[calib]` temperature `t = 1.0767`; Seeds-SSE `1634.2047 → 1627.5445`
- **Category totals**: EL 25.7%, SPECS **26.1%** (target 8% - WORSE with calibration!)
- **Leaderboard score: 4.5042**

**NOTE**: Category calibration made SPECS balance WORSE (21% → 26%). See RUN_ANALYSIS_OCT2.md for details.

## Reproduce Previous Best (~5.03) [Archived]

```powershell
$env:GITHUB_TOKEN = "<YOUR_TOKEN>"
python script.py --use_features --fetch_metrics --kfold 8 --verbose --outdir . --seeds_boost 1.7 --no_calibrate_cats
```

Expected logs:
- CV-picked `lam_z ≈ 0.8`, `lam_theta ≈ 0.3`, `p ≈ 0.0`
- Ensemble weights ~ `seeds_only_ridge: 1.0`
- Temperature `t ≈ 1.07`
- Final Seeds-only SSE ≈ 562.15 (MSE ≈ 3.346)

## Phase 2: Mixed-Effects Model (Oct 4, 2025)

### H1 - Mixed-Effects Bradley-Terry with Juror Category Biases

**Command:**
```bash
python phase2_mixed_effects.py --cv --verbose --outdir .
```

**Model:**
```
log_ratio[j,a,b] = (z_base[b] - z_base[a]) + (β_j[cat(b)] - β_j[cat(a)])
```
- 45 base weights + 222 juror-category biases (37 jurors × 6 categories)
- Regularization: λ_base=3.0, λ_beta=0.5 (CV-selected)
- Alternating least squares optimization (~20 iterations)

**Results:**
- Training: Base-only MSE 4.3481 → Mixed-effects MSE 2.2060 (**49.27% improvement**)
- Learned biases align with juror profiles (e.g., L1Juror21 +1.95 for EL matches 83% win rate)
- **Leaderboard: 4.4753** (0.64% improvement from 4.5042)

**Interpretation:**
- Strong training improvement validates juror heterogeneity exists
- Minimal LB improvement indicates private LB has mostly **new jurors** not in training
- Model correctly falls back to base weights for unknown jurors (β_new = 0)
- Slight improvement from better base weights learned via hierarchical structure

**Key Learning:**
- Private leaderboard juror distribution differs significantly from training
- Juror-specific modeling helps but doesn't generalize to completely new jurors
- Hierarchical approach is sound but limited by juror overlap

---

## Phase 3: Minimal Tweaks Approach (Oct 5, 2025)

### H2 - Conservative Foundational Library Boosts

**Command:**
```bash
python phase3_advanced.py --outdir .
```

**Model:**
- Base Bradley-Terry with **minimal** foundational library boosts (1.15x-1.25x)
- Slight category rebalancing: TOOLS 22% (+10%), LANG 10% (+25%), EL 36% (-5%), CL 21% (-5%)
- Regularization: λ=0.8 (trust training data more than extreme priors)
- Ensemble: 90% base + 5% dev-centric + 5% decentralization

**Foundational Boosts Applied:**
- openzeppelin-contracts: 1.25x (was 25x in aggressive attempt - too high!)
- ethers.js: 1.20x (was 20x - too high!)
- hardhat: 1.15x (was 18x - too high!)
- foundry: 1.20x (was 18x - too high!)
- solidity: 1.15x (was 15x - too high!)

**Final Weights:**
- geth: 17.67%, eips: 13.94%, solidity: 10.06%
- ethers.js: 4.23% (+6% vs baseline)
- openzeppelin: 3.65% (+4% vs baseline)
- foundry: 3.45% (+15% vs baseline)

**Results:**
- **Leaderboard: 4.4460** ← **NEW BEST!**
- 0.6% improvement from Phase 2 (4.4753)
- 1.3% improvement from OSO baseline (4.5042)
- 6.5% improvement from initial baseline (~4.75)

**Critical Learning:**
- **Aggressive approach FAILED**: λ=5.0 with 20x-25x boosts → score 5.21 (WORSE!)
- **Conservative approach WORKS**: λ=0.8 with 1.15x-1.25x boosts → score 4.45 (BETTER!)
- **Training data knows best**: Deviating too far from what training suggests hurts performance
- **Tiny nudges > big assumptions**: 1.2x boost to ethers.js works, 20x boost fails
- The model is well-optimized at baseline; only minor improvements possible

**Interpretation:**
- The baseline was already near-optimal for this dataset
- Foundational libraries DO deserve slightly more weight (validated by +6-15% improvements)
- But extreme reweighting based on theory (juror quotes) doesn't transfer to private LB
- This suggests private LB either: (1) has different juror preferences, or (2) our juror analysis was overfitted to vocal minority

---

## Next Experiment Matrix (Proposed)

- **Phase 3 Enhancements** (if pursuing sub-1.0 on similar data):
  - Sparse repo-specific interactions for frequent comparisons
  - Text mining from juror reasoning fields
  - Temporal dynamics (submission timing effects)
  
- **Ensemble Approaches** (for current competition):
  - Weighted blend: 70% baseline (4.5042) + 30% Phase 2 (4.4753)
  - Might achieve 4.49-4.50 score
  
- **Robustness vs. Caps** (baseline improvements):
  - Huber IRLS (`delta` in [0.8, 1.2, 1.6]) vs. no caps; compare to mild caps (6.0–7.0).
- **Per-Juror Scales**
  - `lam_t` in [0.2, 0.4, 0.6], `t_clip` in [(0.8,1.2), (0.75,1.25)]; evaluate on seeds-only CV folds.
- **Seeds Emphasis CV**
  - Cross-validate `--seeds_boost` in [1.4, 1.5, 1.6, 1.7, 1.8] on seeds-only folds.
- **External Signals**
  - Add release cadence (last 12m), issue/PR closure rate, stars/forks growth, topic density; retrain `theta` and re-run CV.
- **Meta-Ensemble**
  - Include candidates: seeds-only trained robust, juror-scale, prior-only; add small L2 on ensemble weights.
- **Category Calibration Guardrails**
  - Re-enable only if seeds-only CV improves post-temperature; try small L2/L1 and early stopping.

## Interpretation Tips

- Prefer the run with the lowest seeds-only MSE; it’s the best offline correlate—but verify with small A/B leaderboard submissions.
- Watch `[final] category totals` to catch runaway mass (e.g., LANG dominating unexpectedly).
- Track seed list order (`dataset/test.csv`) is preserved; submission weights always sum to 1 across 45 seeds.

## Changelog
- **2025-10-04**: Multiple CV-lite runs with private-LB proxy and no Laplacians. Best leaderboard 4.507 (non-ensemble, seeds-only robust). Additional runs at 4.5124 and 4.5207. Conclusion: no significant improvement beyond ~4.50 without structural/data changes.
- **2025-10-03**: Introduced private-LB proxy into CV. Initial settings with caps/shrink and strong Laplacians regressed to 6.0030. Removing caps/shrink and increasing proxy to 1.5 improved to 4.5225. Weakening Laplacian (0.15) and removing category Laplacian scored 4.5420.

- **2025-10-02**: F1 (4.5042) - Successfully integrated OSO funding + developer boosts (completing the OSO integration). ethers.js now gets 7.01x boost (depRank 2.74x × funding 1.64x × deps 1.56x). Seeds-SSE improved to 1627.5. **CRITICAL FINDING**: Category calibration makes SPECS balance WORSE (21% → 26%) by applying positive offset when negative needed. Recommendation: Keep `--no_calibrate_cats` until eips/solidity concentration fixed at root cause. See RUN_ANALYSIS_OCT2.md.
- **2025-10-01**: Implemented Phase 1 fixes: (1) URL migrations for argotorg→ethereum, (2) OSO funding boost ($2.7M for geth), (3) OSO developer boost (12 devs for geth), (4) Expanded Huber delta grid to [2.0, 3.0, 4.0, 5.0], (5) Blended OSO priors into feature learning (70% learned + 30% OSO). Lambda grid expanded to include stronger regularization [0.02...3.0].
- **2025-09-30**: MAJOR UPDATE after 4.86 → 7.03 score regression. Root cause analysis revealed NEW jurors (50% of expanded training data) value foundational libraries differently. Implemented 3 priority fixes: (1) OSO dependency rank as STRONG prior (was 6x, now conservative 2x for 0.8+ rank), (2) Dependency graph count boost (alloy has 89 dependencies!), (3) Updated category shares (TOOLS 13%→20%, SPECS 3%→8%). Also increased Huber delta grid to preserve extreme multipliers (19.3% of NEW data has 100+x). See DIAGNOSIS.md and IMPLEMENTATION_SUMMARY.md for details.
- 2025-09-29: Added E2 (4.8916), E3 (4.9279), E4 (n/s); documented that E3/E4 worsened outcomes; reverted to previously best-performing script per user.
- 2025-09-25: Added tuned seeds_boost & robust CV; new current best 4.8827 with ensemble and temperature details.
- 2025-09-25: Initial compilation with best leaderboard ≈ 5.0295 and contrasting runs.
