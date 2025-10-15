# Deep Funding L1 - Ethereum Dependency Allocation

**Competition:** Deep Funding L1 - Quantifying Ethereum Open Source Dependencies  
**Best Submission:** Phase 3 (Private LB: 6.4588)  
**Date:** October 2025

---

## Quick Start

### Prerequisites
- Python 3.9+
- NumPy 1.24+

### Run Best Model (Phase 3)

```bash
python phase3_advanced.py --outdir .
```

**Output:** `submission_phase3_YYYYMMDD_HHMMSS.csv` with weights for 45 seed repositories.

### Command-Line Options

```bash
# With verbose logging
python phase3_advanced.py --verbose --outdir .

# Custom regularization strength (default: 0.8)
python phase3_advanced.py --lambda_reg 0.8 --outdir .

# Custom ensemble weights (base, dev-centric, decentralization)
python phase3_advanced.py --ensemble_weights 0.9 0.05 0.05 --outdir .
```

---

## Repository Structure

### Core Files

```
phase3_advanced.py          # Best model (Phase 3, score: 6.46 private)
script.py                   # Utilities (categorization, URL normalization)
seedRepos.json              # 45 seed repositories
dataset/
  └── train.csv             # 407 pairwise juror comparisons
dependency-graph-main/
  └── datasets/pairwise/
      └── getProjectMetadata.json  # OSO metrics (dependency rank, funding, devs)
```

### Submissions

```
submission_phase3_20251005_145132.csv   # Best submission (6.4588 private)
```

### Documentation

```
WRITEUP_FINAL.md            # Full methodology and results writeup
experiments.md              # Detailed experiment log
README.md                   # This file
```

### Visualizations

```
viz_1_score_evolution.png       # Score journey chart
viz_2_prediction_vs_actual.png  # Prediction accuracy scatter
viz_3_approach_comparison.png   # Phase comparison heatmap
viz_4_boost_impact.png          # Boost strategy comparison
create_visualizations.py        # Script to generate charts
```

### Analysis Scripts (Optional)

```
analyze_jurors.py           # Juror preference analysis
analyze_comprehensive.py    # Category distribution analysis
analyze_data.py             # Data quality checks
```

---

## Data Requirements

### Input Data

**1. Training Comparisons** (`dataset/train.csv`)
- 407 pairwise comparisons from 37 expert jurors
- Format: `repo_a, repo_b, choice, multiplier, juror_id`
- Example: `ethereum/go-ethereum, nethermindeth/nethermind, 1, 1.5, L1Juror1`

**2. Seed Repositories** (`seedRepos.json`)
- 45 core Ethereum repositories
- Format: JSON array of GitHub URLs
- Example: `["https://github.com/ethereum/go-ethereum", ...]`

**3. OSO Metrics** (`dependency-graph-main/datasets/pairwise/getProjectMetadata.json`)
- Open Source Observer ecosystem metrics
- Keys used: `osoDependencyRank`, `totalFundingUsd`, `avgFullTimeDevs`
- Source: https://www.opensource.observer/

### Output Format

**Submission CSV:**
```csv
repo,parent,weight
https://github.com/ethereum/go-ethereum,ethereum,0.1767
https://github.com/ethereum/eips,ethereum,0.1394
...
```

- All weights sum to 1.0
- One row per seed repository
- Parent is always "ethereum"

---

## Model Overview

### Phase 3: Conservative Approach (BEST)

**Algorithm:** Bradley-Terry in log-space with OSO-informed priors

**Key Features:**
- OSO dependency rank as primary signal (not GitHub stars)
- Minimal foundational boosts (1.15× - 1.25×)
- Slight category rebalancing (TOOLS +10%, LANG +25%)
- Conservative regularization (λ=0.8, trust training data)
- Ensemble: 90% base + 5% dev-centric + 5% decentralization

**Results:**
- Public LB: 4.4460 (best)
- Private LB: 6.4588 (best)

### Other Phases (for comparison)

**Phase 2:** Complex juror-specific model (Private: 6.6637, +3.2% worse)  
**Phase 5:** Constrained optimization (Private: 6.7476, +4.5% worse)

---

## Reproducing Results

### Step 1: Verify Data

```bash
# Check files exist
ls dataset/train.csv
ls seedRepos.json
ls dependency-graph-main/datasets/pairwise/getProjectMetadata.json
```

### Step 2: Run Phase 3

```bash
python phase3_advanced.py --verbose --outdir .
```

### Step 3: Expected Output

```
[1] Loading data...
  Training samples: 407
  Seed repos: 45
  OSO coverage: 43/45

[2] Building enhanced priors...
  TOOLS: 22.00%
  LANG:  10.00%
  EL:    36.00%
  CL:    21.00%
  SPECS:  8.00%
  INFRA:  3.00%

[3] Creating archetype ensembles...
  Ensemble weights: Base=0.90, DevCentric=0.05, Decentralization=0.05

[6] Final weights (top 15):
  1. ethereum/go-ethereum          17.67%
  2. ethereum/eips                 13.94%
  3. argotorg/solidity             10.06%
  ...

[OK] Submission saved to: submission_phase3_YYYYMMDD_HHMMSS.csv
```

### Step 4: Generate Visualizations (Optional)

```bash
python create_visualizations.py
```

Generates 4 PNG charts showing score evolution, predictions vs actual, approach comparison, and boost impact.

---

## Key Insights

### What Worked
✅ Trust training data (λ=0.8 regularization)  
✅ OSO metrics > GitHub metrics (dependency rank captures foundational nature)  
✅ Conservative boosts (1.2×) > Aggressive boosts (20×)  
✅ Simple models > Complex juror-specific models  

### What Failed
❌ Juror-specific category biases (overfits to known jurors)  
❌ Hard constraints on weights (fights against correct direction)  
❌ Aggressive foundational boosts (causes instability)  
❌ Domain expertise over data (my intuitions were wrong)  

---

## Dependencies

### Python Packages

```
numpy>=1.24.0
```

No other dependencies required for Phase 3.

**Optional (for analysis scripts):**
```
matplotlib>=3.5.0  # For visualizations
pandas>=1.5.0      # For data analysis
```

### Data Files

All required data files are included in this repository:
- `dataset/train.csv` (407 samples, 45 repos, 37 jurors)
- `seedRepos.json` (45 seed repositories)
- `dependency-graph-main/datasets/pairwise/getProjectMetadata.json` (OSO metrics)

**No external API calls required** - all data is local.

---

## License

Open source - all code is available in this repository for educational and research purposes.

---

## Contact

For questions about implementation, see code comments in `phase3_advanced.py`.

---

## Acknowledgments

- **Deep Funding** organizers for the competition
- **Open Source Observer** for ecosystem metrics
- **Ethereum Foundation** for supporting open source funding research
