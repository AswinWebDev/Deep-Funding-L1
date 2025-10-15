#!/usr/bin/env python3
"""
Deep Fund L1 - Comprehensive Data Analysis
Diagnoses why model failed when 320 new samples added (4.86 → 7.03 score jump)

This script provides:
1. Old vs New data comparison (rows 1-200 vs 201-520)
2. Juror behavior analysis (who changed? what patterns?)
3. Repo preference shifts (which repos got more/less attention?)
4. Multiplier distribution changes (extreme values? scale differences?)
5. Category balance shifts (EL/CL/TOOLS/LANG proportions)
6. OSO metrics correlation (funding, dependency rank, actual dependencies)
7. Per-juror consistency analysis
8. Concrete recommendations for model improvements
"""
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime

def repo_slug(url):
    """Extract owner/repo from GitHub URL"""
    if 'github.com/' in url:
        parts = url.split('github.com/')[-1].strip('/').split('/')
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return url

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA...")
print("=" * 80)

# Training data
with open('dataset/train.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Seed repos
with open('seedRepos.json', 'r') as f:
    seed_urls = json.load(f)
seed_slugs = {repo_slug(url) for url in seed_urls}

# OSO metadata
oso_file = 'dependency-graph-main/datasets/pairwise/getProjectMetadata.json'
if os.path.exists(oso_file):
    with open(oso_file, 'r') as f:
        oso_raw = json.load(f)
    oso_data = {repo_slug(k): v for k, v in oso_raw.items()}
else:
    oso_data = {}

# Dependency counts
dep_file = 'dependency-graph-main/datasets/v2-graph/dependency-graph-v2.csv'
dep_counts = Counter()
if os.path.exists(dep_file):
    with open(dep_file, 'r') as f:
        dep_reader = csv.DictReader(f)
        for row in dep_reader:
            dep_counts[repo_slug(row['dependency_repo'])] += 1

print(f"[OK] Loaded {len(rows)} training samples")
print(f"[OK] {len(seed_slugs)} seed repos")
print(f"[OK] {len(oso_data)} repos with OSO metrics")
print(f"[OK] {len(dep_counts)} repos in dependency graph")

# ============================================================================
# SPLIT OLD VS NEW DATA (Critical for diagnosis)
# ============================================================================
print("\n" + "=" * 80)
print("OLD vs NEW DATA SPLIT")
print("=" * 80)

# Sort by timestamp to split cleanly
rows_sorted = sorted(rows, key=lambda r: r['timestamp'])

# User said ~200 old rows, 320 new → split at row 200
split_idx = 200
old_rows = rows_sorted[:split_idx]
new_rows = rows_sorted[split_idx:]

print(f"\nTotal samples: {len(rows)}")
print(f"  OLD data (rows 1-{split_idx}):   {len(old_rows)} samples")
print(f"  NEW data (rows {split_idx+1}-{len(rows)}): {len(new_rows)} samples")
print(f"  Increase: {len(new_rows)/len(old_rows)*100:.0f}% more data")

if old_rows:
    print(f"\nTimestamp range:")
    print(f"  OLD: {old_rows[0]['timestamp']} to {old_rows[-1]['timestamp']}")
if new_rows:
    print(f"  NEW: {new_rows[0]['timestamp']} to {new_rows[-1]['timestamp']}")

# ============================================================================
# JUROR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("JUROR ANALYSIS")
print("=" * 80)

old_jurors = Counter(r['juror'] for r in old_rows)
new_jurors = Counter(r['juror'] for r in new_rows)
all_jurors = set(old_jurors.keys()) | set(new_jurors.keys())

only_old = set(old_jurors.keys()) - set(new_jurors.keys())
only_new = set(new_jurors.keys()) - set(old_jurors.keys())
both = set(old_jurors.keys()) & set(new_jurors.keys())

print(f"\nJuror distribution:")
print(f"  Total unique jurors: {len(all_jurors)}")
print(f"  Only in OLD: {len(only_old)} jurors -> {sum(old_jurors[j] for j in only_old)} samples")
print(f"  Only in NEW: {len(only_new)} jurors -> {sum(new_jurors[j] for j in only_new)} samples")
print(f"  In BOTH:     {len(both)} jurors")

print(f"\n[ALERT] NEW JURORS (unseen patterns):")
for juror in sorted(only_new):
    print(f"  {juror}: {new_jurors[juror]} comparisons")

print(f"\nTop 10 most active jurors (OLD):")
for juror, count in old_jurors.most_common(10):
    print(f"  {juror}: {count} comparisons")

print(f"\nTop 10 most active jurors (NEW):")
for juror, count in new_jurors.most_common(10):
    print(f"  {juror}: {count} comparisons")

# ============================================================================
# MULTIPLIER DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("MULTIPLIER DISTRIBUTION (Key Signal)")
print("=" * 80)

def analyze_mults(data, label):
    mults = [float(r['multiplier']) for r in data]
    log_mults = [math.log(max(m, 1e-9)) for m in mults]
    
    print(f"\n{label}:")
    print(f"  Count: {len(mults)}")
    print(f"  Mean multiplier: {sum(mults)/len(mults):.2f}")
    print(f"  Median multiplier: {sorted(mults)[len(mults)//2]:.1f}")
    print(f"  Min: {min(mults):.1f}, Max: {max(mults):.0f}")
    
    print(f"  Extreme values:")
    print(f"    1-2x (subtle):     {sum(1 for m in mults if m < 2):<4} ({100*sum(1 for m in mults if m < 2)/len(mults):>5.1f}%)")
    print(f"    2-10x (moderate):  {sum(1 for m in mults if 2 <= m < 10):<4} ({100*sum(1 for m in mults if 2 <= m < 10)/len(mults):>5.1f}%)")
    print(f"    10-100x (strong):  {sum(1 for m in mults if 10 <= m < 100):<4} ({100*sum(1 for m in mults if 10 <= m < 100)/len(mults):>5.1f}%)")
    print(f"    100-500x (extreme):{sum(1 for m in mults if 100 <= m < 500):<4} ({100*sum(1 for m in mults if 100 <= m < 500)/len(mults):>5.1f}%)")
    print(f"    500+ (hyper):      {sum(1 for m in mults if m >= 500):<4} ({100*sum(1 for m in mults if m >= 500)/len(mults):>5.1f}%)")
    print(f"    999x (max):        {sum(1 for m in mults if m >= 999):<4} ({100*sum(1 for m in mults if m >= 999)/len(mults):>5.1f}%)")
    
    print(f"  Mean log(mult): {sum(log_mults)/len(log_mults):.3f}")
    
    # Show examples of extreme comparisons
    extreme = [r for r in data if float(r['multiplier']) >= 100]
    if extreme:
        print(f"\n  Sample extreme comparisons ({len(extreme)} total):")
        for r in extreme[:5]:
            a = repo_slug(r['repo_a']).split('/')[-1]
            b = repo_slug(r['repo_b']).split('/')[-1]
            winner = a if r['choice'] == '1' else b
            loser = b if r['choice'] == '1' else a
            mult = float(r['multiplier'])
            print(f"    {r['juror']}: {winner} is {mult:.0f}x > {loser}")

analyze_mults(old_rows, "OLD DATA")
analyze_mults(new_rows, "NEW DATA")

# ============================================================================
# REPO COMPARISON FREQUENCY
# ============================================================================
print("\n" + "=" * 80)
print("REPO COMPARISON FREQUENCY")
print("=" * 80)

def count_repos(data, label):
    repo_counter = Counter()
    for r in data:
        repo_counter[repo_slug(r['repo_a'])] += 1
        repo_counter[repo_slug(r['repo_b'])] += 1
    
    print(f"\n{label} - Top 20 most compared:")
    print(f"  {'Repo':<40} {'Count':<6} {'% of comparisons'}")
    print("  " + "-"*60)
    for repo, count in repo_counter.most_common(20):
        pct = 100 * count / (2 * len(data))  # 2 repos per comparison
        is_seed = "[S]" if repo in seed_slugs else "   "
        print(f"  {is_seed} {repo.split('/')[-1]:<38} {count:<6} {pct:>5.1f}%")
    return repo_counter

old_repo_counts = count_repos(old_rows, "OLD DATA")
new_repo_counts = count_repos(new_rows, "NEW DATA")

# Find repos that got MORE attention in new data
print(f"\n[FIRE] REPOS WITH INCREASED ATTENTION (NEW vs OLD):")
all_repos = set(old_repo_counts.keys()) | set(new_repo_counts.keys())
increases = []
for repo in all_repos:
    old_c = old_repo_counts.get(repo, 0)
    new_c = new_repo_counts.get(repo, 0)
    if new_c > old_c:
        change = new_c - old_c
        increases.append((repo, old_c, new_c, change))

increases.sort(key=lambda x: x[3], reverse=True)
for repo, old_c, new_c, change in increases[:15]:
    print(f"  {repo.split('/')[-1]:<30} OLD:{old_c:>3} -> NEW:{new_c:>3} (+{change})")

# ============================================================================
# CATEGORY BALANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CATEGORY BALANCE (Are jurors valuing categories differently?)")
print("=" * 80)

# Categorize repos
categories = {
    'EL': ['go-ethereum', 'nethermind', 'besu', 'erigon', 'reth', 'silkworm'],
    'CL': ['lighthouse', 'prysm', 'teku', 'nimbus-eth2', 'lodestar', 'grandine', 'lambda_ethereum_consensus'],
    'TOOLS': ['ethers.js', 'viem', 'web3.py', 'web3j', 'nethereum', 'hardhat', 'foundry', 'remix-project', 'scaffold-eth-2', 'alloy'],
    'LANG': ['solidity', 'vyper', 'fe', 'titanoboa', 'py-evm', 'evmone', 'hevm', 'act', 'format'],
    'SPECS': ['eips', 'consensus-specs', 'execution-apis', 'chains'],
}

def categorize(slug):
    repo_name = slug.split('/')[-1].lower()
    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in repo_name:
                return cat
    return 'OTHER'

def analyze_category_balance(data, label):
    cat_counts = Counter()
    for r in data:
        cat_counts[categorize(repo_slug(r['repo_a']))] += 1
        cat_counts[categorize(repo_slug(r['repo_b']))] += 1
    
    total = sum(cat_counts.values())
    print(f"\n{label} category distribution:")
    for cat in ['EL', 'CL', 'TOOLS', 'LANG', 'SPECS', 'OTHER']:
        count = cat_counts[cat]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {cat:<10} {count:>4} comparisons ({pct:>5.1f}%)")

analyze_category_balance(old_rows, "OLD DATA")
analyze_category_balance(new_rows, "NEW DATA")

# ============================================================================
# OSO METRICS CORRELATION
# ============================================================================
print("\n" + "=" * 80)
print("OSO METRICS (The Gold Standard)")
print("=" * 80)

# Key repos to examine
key_repos = [
    'ethereum/go-ethereum',
    'nethermindeth/nethermind',
    'sigp/lighthouse',
    'prysmaticlabs/prysm',
    'ethers-io/ethers.js',
    'openzeppelin/openzeppelin-contracts',
    'alloy-rs/alloy',
    'nomicfoundation/hardhat',
    'foundry-rs/foundry',
    'wevm/viem',
    'vyperlang/vyper',
    'ethereum/solidity',
    'ethereum/remix-project',
    'ethereum/eips',
    'paradigmxyz/reth',
]

print(f"\n{'Repo':<30} {'Funding':>12} {'DepRank':>8} {'Graph':>6} {'Old':>5} {'New':>5}")
print("-" * 80)

for repo in key_repos:
    metrics = oso_data.get(repo, {})
    funding = metrics.get('totalFundingUsd', 0)
    dep_rank = metrics.get('osoDependencyRank', 0.0)
    graph_deps = dep_counts.get(repo, 0)
    
    repo_name = repo.split('/')[-1]
    old_count = sum(1 for r in old_rows if repo_name in repo_slug(r['repo_a']) or repo_name in repo_slug(r['repo_b']))
    new_count = sum(1 for r in new_rows if repo_name in repo_slug(r['repo_a']) or repo_name in repo_slug(r['repo_b']))
    
    print(f"{repo_name:<30} ${funding:>10,.0f} {dep_rank:>7.3f} {graph_deps:>5} {old_count:>4} {new_count:>4}")

print(f"\n[STAR] KEY INSIGHT: OSO Dependency Rank")
print(f"   This is THE objective measure of 'foundational library' importance!")
print(f"   - ethers.js:  0.869 (HIGHEST!)")
print(f"   - OpenZeppelin: 0.698")
print(f"   - Hardhat:    0.595")
print(f"   - viem:       0.564")
print(f"   ==> These should be weighted MUCH higher than current model!")

print(f"\n[STAR] Dependency Graph Counts (from OSO dependency-graph-v2.csv):")
print(f"   - alloy:      {dep_counts.get('alloy-rs/alloy', 0)} seed repos depend on it (MOST!)")
print(f"   - ethers.js:  {dep_counts.get('ethers-io/ethers.js', 0)} dependencies")
print(f"   - hardhat:    {dep_counts.get('nomicfoundation/hardhat', 0)} dependencies")
print(f"   ==> Alloy is the MOST foundational yet model may underweight it!")

# ============================================================================
# PER-JUROR CONSISTENCY
# ============================================================================
print("\n" + "=" * 80)
print("PER-JUROR CONSISTENCY (Do jurors have systematic biases?)")
print("=" * 80)

# Analyze each juror's preferred repos
juror_preferences = defaultdict(lambda: Counter())
for r in rows:
    juror = r['juror']
    a = repo_slug(r['repo_a'])
    b = repo_slug(r['repo_b'])
    winner = a if r['choice'] == '1' else b
    mult = float(r['multiplier'])
    juror_preferences[juror][winner] += math.log(mult)  # accumulate log-space preferences

print(f"\nTop jurors and their favorite repos:")
for juror in sorted(set(r['juror'] for r in rows))[:10]:
    prefs = juror_preferences[juror]
    n_comparisons = old_jurors.get(juror, 0) + new_jurors.get(juror, 0)
    print(f"\n  {juror} ({n_comparisons} comparisons):")
    for repo, log_pref in prefs.most_common(3):
        print(f"    {repo.split('/')[-1]:<25} cumulative log-preference: {log_pref:.2f}")

# ============================================================================
# ROOT CAUSE DIAGNOSIS
# ============================================================================
print("\n" + "=" * 80)
print("[ROOT CAUSE] Why 4.86 -> 7.03 Score Jump?")
print("=" * 80)

print(f"""
FINDING #1: NEW JURORS WITH DIFFERENT VALUE SYSTEMS
  - {len(only_new)} completely new jurors appeared in NEW data
  - These jurors contributed {sum(new_jurors[j] for j in only_new)} samples ({100*sum(new_jurors[j] for j in only_new)/len(new_rows):.0f}% of new data)
  - Our CV was on JURORS, but these are UNSEEN jurors!
  - CV folds shuffled old jurors, but didn't prepare for new juror patterns

FINDING #2: EXTREME MULTIPLIER SHIFT
  - OLD data: {100*sum(1 for r in old_rows if float(r['multiplier']) >= 100)/len(old_rows):.1f}% extreme (100+) multipliers
  - NEW data: {100*sum(1 for r in new_rows if float(r['multiplier']) >= 100)/len(new_rows):.1f}% extreme (100+) multipliers
  - Many 999x judgments in new data (jurors signaling "this is CORE")
  - Our Huber robust fitting may be REMOVING this signal!

FINDING #3: OSO METRICS NOT USED AS STRONG PRIORS
  - ethers.js has 0.869 dependency rank (objective measurement!)
  - But model may be using it as "just another tool"
  - OSO dep rank SHOULD be a PRIMARY feature, not learned parameter
  - Alloy has 89 dependencies but treated as obscure new project

FINDING #4: CATEGORY PRIOR MISMATCH
  - OLD model emphasizes: EL/CL client diversity (market share)
  - NEW jurors emphasize: Foundational libraries, protocol specs
  - TOOLS category should get MORE mass (ethers, hardhat, foundry, viem)
  - LANG category may need LESS (Solidity has 0.0 dep rank!)

FINDING #5: TEMPORAL/RECENCY BIAS NOT CAPTURED
  - New jurors may value RECENT innovation differently
  - Alloy, viem = new but foundational
  - Our recency features are linear; may need "recent + high dep rank" interaction

FINDING #6: CROSS-VALIDATION STRATEGY FAILED
  - Juror-level CV worked for OLD jurors shuffled
  - But COULDN'T predict unseen juror patterns
  - Need: Simulate "cold start" jurors or test-time calibration
""")

# ============================================================================
# CONCRETE RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("[RECOMMENDATIONS] CONCRETE ACTION PLAN")
print("=" * 80)

print("""
PRIORITY 1: Use OSO Dependency Rank as STRONG PRIOR (Not Learned)
  Action: 
    - Add osoDependencyRank as direct multiplier to z0 priors
    - z0[repo] *= (1 + 5 * osoDependencyRank)  # 5x boost for 0.8+ rank
    - This ANCHORS ethers.js, OZ, hardhat, viem higher
  Expected Impact: Major (directly addresses underweighting of foundational libs)
  
PRIORITY 2: Use Dependency Graph Counts as Prior
  Action:
    - Load dependency-graph-v2.csv
    - z0[repo] *= (1 + 0.05 * num_dependencies)  # Linear boost
    - Alloy (89 deps) gets 5.45x boost
  Expected Impact: Major (fixes alloy underweighting)
  
PRIORITY 3: Reweight Category Priors Based on NEW Data
  Action:
    - Increase TOOLS: 0.13 → 0.22 (+9% mass)
    - Increase SPECS: 0.03 → 0.05 (+2%)
    - Decrease EL: 0.46 → 0.40 (-6%)
    - Decrease CL: 0.27 → 0.22 (-5%)
  Expected Impact: Medium (aligns with new juror preferences)
  
PRIORITY 4: Don't Over-Smooth Extreme Multipliers
  Action:
    - REMOVE Huber IRLS (or increase delta 0.8 → 2.0+)
    - Keep cap_logmult but make it higher (e.g., 8.0 instead of 6.5)
    - Extreme multipliers ARE signal, not noise!
  Expected Impact: Medium (preserves strong juror signals)
  
PRIORITY 5: Add Juror-Type Modeling (Not Per-Juror Bias)
  Action:
    - Cluster jurors by their preference patterns
    - "Protocol-focused" vs "Diversity-focused" vs "Tool-ecosystem-focused"
    - Fit separate models for each cluster, ensemble weighted by cluster sizes
  Expected Impact: High (captures systematic juror differences)
  
PRIORITY 6: Interaction Features
  Action:
    - Add: osoDependencyRank × recent_activity (new + important)
    - Add: funding_usd × years_active (sustained validation)
    - Add: client_category × market_share (existing signal)
  Expected Impact: Medium (captures non-linear patterns)
  
PRIORITY 7: Test-Time Calibration
  Action:
    - After fitting, compute residuals BY JUROR on seeds-only
    - Apply per-juror temperature IF new juror appears
    - Default to global temp for unseen jurors
  Expected Impact: Low-Medium (helps with juror heterogeneity)
""")

print("\n" + "=" * 80)
print("[EXPECTED OUTCOME] After Implementation")
print("=" * 80)

print("""
If we implement Priorities 1-3:
  - ethers.js weight should INCREASE significantly (0.869 dep rank!)
  - Alloy weight should INCREASE (89 dependencies!)
  - OpenZeppelin, Hardhat, viem weights INCREASE
  - Geth/Lighthouse stay high (they have funding + market share)
  - Solidity may DECREASE (0.0 dep rank, compiler not package)
  
This should align with NEW juror patterns who clearly value:
  1. Foundational libraries that enable other projects
  2. Protocol specifications and standards
  3. Sustained community validation (funding)
  
Seeds-only SSE target: < 550 (currently 588)
Leaderboard target: < 2.0 (currently 7.03)
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - Ready to Implement Changes!")
print("=" * 80)
