#!/usr/bin/env python3
"""
Deep Fund L1 - Comprehensive Data Analysis
Diagnoses why model failed when 320 new samples added (4.86 → 7.03 score jump)
"""
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

print("Loading training data...")
# Load training data
with open('dataset/train.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print("="*80)
print("DATA COMPOSITION ANALYSIS")
print("="*80)

# Split old vs new
old_cutoff = '2025-03-30'
old_rows = [r for r in rows if r['timestamp'] < old_cutoff]
new_rows = [r for r in rows if r['timestamp'] >= old_cutoff]

print(f"\nTotal rows: {len(rows)}")
print(f"  OLD data (before {old_cutoff}): {len(old_rows)} rows ({100*len(old_rows)/len(rows):.1f}%)")
print(f"  NEW data ({old_cutoff}+):         {len(new_rows)} rows ({100*len(new_rows)/len(rows):.1f}%)")

# Juror analysis
old_jurors = set(r['juror'] for r in old_rows)
new_jurors = set(r['juror'] for r in new_rows)
only_new_jurors = new_jurors - old_jurors

print(f"\nJuror counts:")
print(f"  Total unique jurors: {len(old_jurors | new_jurors)}")
print(f"  OLD data jurors: {len(old_jurors)}")
print(f"  NEW data jurors: {len(new_jurors)}")
print(f"  Only in NEW: {len(only_new_jurors)} jurors")
print(f"    {sorted(only_new_jurors)}")

# Multiplier analysis
def analyze_multipliers(data_rows, label):
    mults = [float(r['multiplier']) for r in data_rows]
    print(f"\n{label} Multiplier Stats:")
    print(f"  Count: {len(mults)}")
    print(f"  Mean: {sum(mults)/len(mults):.1f}")
    print(f"  Median: {sorted(mults)[len(mults)//2]:.1f}")
    print(f"  Max: {max(mults):.0f}")
    print(f"  >=100: {sum(1 for m in mults if m >= 100)} samples ({100*sum(1 for m in mults if m >= 100)/len(mults):.1f}%)")
    print(f"  >=500: {sum(1 for m in mults if m >= 500)} samples ({100*sum(1 for m in mults if m >= 500)/len(mults):.1f}%)")
    extreme_999 = [r for r in data_rows if float(r['multiplier']) == 999]
    if extreme_999:
        print(f"  999x multipliers: {len(extreme_999)} (extreme preference signal)")
        for r in extreme_999[:3]:
            a = r['repo_a'].split('/')[-1]
            b = r['repo_b'].split('/')[-1]
            choice = a if r['choice'] == '1' else b
            print(f"    {r['juror']}: {choice} >>> other")

analyze_multipliers(old_rows, "OLD")
analyze_multipliers(new_rows, "NEW")

# Repo comparison frequency
print("\n" + "="*80)
print("REPO COMPARISON PATTERNS")
print("="*80)

def get_repo_counts(data_rows, label):
    repo_counter = Counter()
    for r in data_rows:
        repo_counter[r['repo_a'].split('/')[-1]] += 1
        repo_counter[r['repo_b'].split('/')[-1]] += 1
    print(f"\n{label} - Top 15 most compared repos:")
    for repo, count in repo_counter.most_common(15):
        print(f"  {repo:30s} {count:3d} comparisons")
    return repo_counter

old_repos = get_repo_counts(old_rows, "OLD DATA")
new_repos = get_repo_counts(new_rows, "NEW DATA")

# Load OSO data
print("\n" + "="*80)
print("OSO DEPENDENCY & FUNDING DATA")
print("="*80)

with open('dependency-graph-main/datasets/pairwise/getProjectMetadata.json', 'r') as f:
    oso_data = json.load(f)

# Count dependencies
with open('dependency-graph-main/datasets/v2-graph/dependency-graph-v2.csv', 'r') as f:
    dep_reader = csv.DictReader(f)
    dep_counts = Counter()
    for row in dep_reader:
        dep_counts[row['dependency_repo']] += 1

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
    'argotorg/solidity',
    'ethereum/remix-project',
]

print("\nKey Seed Repos - OSO Metrics:")
print(f"{'Repo':<30} {'Funding':<12} {'DepRank':<8} {'Deps':<6} {'InNewData'}")
print("-"*80)

for repo in key_repos:
    url = f'https://github.com/{repo}'
    if url in oso_data:
        metrics = oso_data[url]
        funding = metrics.get('totalFundingUsd', 0)
        dep_rank = metrics.get('osoDependencyRank', 0)
        num_deps = metrics.get('numDependentsInOso', 0)
        
        # Count in dependency graph
        dep_graph_count = dep_counts.get(url, 0)
        
        # Check if heavily compared in new data
        repo_name = repo.split('/')[-1]
        new_count = new_repos.get(repo_name, 0)
        
        print(f"{repo_name:<30} ${funding:>10,.0f} {dep_rank:>6.3f} {dep_graph_count:>5} {new_count:>6}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find repos with high OSO rank but low model weight
print("\nHigh OSO Dependency Rank (should be heavily weighted):")
print("  ethers.js:    0.869 (HIGHEST!) - 28 seed repos depend on it")
print("  OZ contracts: 0.698 - security foundation")
print("  Hardhat:      0.595 - dev tool ecosystem")
print("  viem:         0.564 - modern TypeScript library")
print("  Alloy:        89 dependencies in graph (MOST FOUNDATIONAL!)")

print("\nHigh Historical Funding (community validation):")
print("  Geth:         $2,657k (highest)")
print("  Nethermind:   $1,937k")
print("  ethers.js:    $1,911k")
print("  Lighthouse:   $1,910k")

print("\nNew data patterns:")
print(f"  - {len(only_new_jurors)} completely new jurors with different preferences")
print(f"  - 21% of new comparisons use 100+ multipliers (vs lower in old)")
print(f"  - Many 999x 'extreme preference' signals from new jurors")
print(f"  - Core protocol repos (EIPs, specs, clients) heavily compared")

print("\n" + "="*80)
print("HYPOTHESIS: Why Score Jumped 4.86 → 7.03")
print("="*80)
print("""
1. NEW JUROR PATTERNS NOT CAPTURED:
   - 11+ new jurors (L1Juror26-40) joined post-March 2025
   - These jurors show different valuation patterns
   - Many use extreme multipliers (999x) for 'core protocol' emphasis

2. MISSING SIGNALS IN FEATURES:
   - OSO Dependency Rank NOT used as strong prior
   - ethers.js has 0.869 dep rank but may be underweighted
   - Alloy has 89 dependencies but treated as 'new project'
   
3. OVER-RELIANCE ON MARKET SHARE PRIORS:
   - EL/CL client shares from 2024 may not match juror values
   - Jurors value 'foundational libraries' differently than 'client usage'
   - Tools category (ethers.js, hardhat, foundry) may be underweighted

4. EXTREME MULTIPLIER HANDLING:
   - 21% of new data uses 100+ multipliers
   - Model may be smoothing these out too much
   - Robust fitting (Huber) might be removing signal

5. CATEGORY MISALIGNMENT:
   - NEW jurors emphasize: Core protocol specs, foundational libs
   - OLD model emphasized: Client diversity, market share
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
