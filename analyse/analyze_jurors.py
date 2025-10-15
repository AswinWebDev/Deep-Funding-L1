#!/usr/bin/env python3
"""
Deep analysis of juror behavior patterns to inform mixed-effects model
"""
import csv
import json
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def repo_slug(url: str) -> str:
    """Extract owner/repo from URL"""
    parts = url.split("github.com/")
    if len(parts) < 2:
        return url
    tail = parts[1].strip("/")
    comps = [c for c in tail.split("/") if c]
    if len(comps) >= 2:
        return f"{comps[0]}/{comps[1]}"
    return tail

def categorize(slug: str) -> str:
    """Simplified categorization"""
    s = slug.lower()
    if any(k in s for k in ["go-ethereum", "nethermind", "besu", "erigon", "reth", "silkworm"]):
        return "EL"
    if any(k in s for k in ["lighthouse", "prysm", "teku", "nimbus", "lodestar", "grandine", "lambda_ethereum"]):
        return "CL"
    if any(k in s for k in ["solidity", "vyper", "titanoboa", "fe", "py-evm", "evmone", "hevm", "act", "format"]):
        return "LANG"
    if any(k in s for k in ["eips", "consensus-specs", "execution-apis", "chains"]):
        return "SPECS"
    if any(k in s for k in ["pandaops", "helm", "checkpoint"]):
        return "INFRA"
    return "TOOLS"

# Load training data
print("Loading training data...")
with open("dataset/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    samples = list(reader)

print(f"Total samples: {len(samples)}")

# Extract juror statistics
jurors = defaultdict(list)
for row in samples:
    juror = row["juror"]
    repo_a = repo_slug(row["repo_a"])
    repo_b = repo_slug(row["repo_b"])
    choice = int(row["choice"])
    mult = float(row["multiplier"])
    
    # Convert to log-ratio (positive if repo_b wins)
    if choice == 1:  # repo_a wins
        log_ratio = -math.log(mult)
        winner = repo_a
        loser = repo_b
    else:  # repo_b wins
        log_ratio = math.log(mult)
        winner = repo_b
        loser = repo_a
    
    jurors[juror].append({
        'repo_a': repo_a,
        'repo_b': repo_b,
        'cat_a': categorize(repo_a),
        'cat_b': categorize(repo_b),
        'log_ratio': log_ratio,
        'multiplier': mult,
        'winner': winner,
        'loser': loser,
        'winner_cat': categorize(winner),
        'loser_cat': categorize(loser),
    })

print(f"\nNumber of jurors: {len(jurors)}")
print(f"Comparisons per juror: min={min(len(v) for v in jurors.values())}, max={max(len(v) for v in jurors.values())}, mean={sum(len(v) for v in jurors.values()) / len(jurors):.1f}")

# Analyze each juror
print("\n" + "="*100)
print("JUROR PROFILES")
print("="*100)

juror_profiles = {}

for juror_id in sorted(jurors.keys()):
    comps = jurors[juror_id]
    n = len(comps)
    
    # Multiplier statistics
    mults = [c['multiplier'] for c in comps]
    log_ratios = [c['log_ratio'] for c in comps]
    
    # Category preferences (which categories do they favor?)
    cat_wins = Counter(c['winner_cat'] for c in comps)
    cat_losses = Counter(c['loser_cat'] for c in comps)
    
    # Category pair patterns
    cat_pairs = Counter((c['cat_a'], c['cat_b']) for c in comps)
    
    # Compute category bias (how often does each category win vs lose?)
    categories = set(cat_wins.keys()) | set(cat_losses.keys())
    cat_ratios = {}
    for cat in categories:
        wins = cat_wins[cat]
        losses = cat_losses[cat]
        if wins + losses > 0:
            cat_ratios[cat] = wins / (wins + losses)
    
    profile = {
        'n_comparisons': n,
        'mult_min': min(mults),
        'mult_max': max(mults),
        'mult_mean': sum(mults) / len(mults),
        'mult_median': sorted(mults)[len(mults)//2],
        'log_ratio_mean': sum(log_ratios) / len(log_ratios),
        'log_ratio_std': (sum((x - sum(log_ratios)/len(log_ratios))**2 for x in log_ratios) / len(log_ratios))**0.5,
        'cat_win_rates': cat_ratios,
        'extreme_comparisons': sum(1 for m in mults if m >= 100),
        'conservative_comparisons': sum(1 for m in mults if m <= 10),
    }
    
    juror_profiles[juror_id] = profile
    
    # Print summary
    if n >= 10:  # Only print jurors with substantial data
        print(f"\n{juror_id} ({n} comparisons)")
        print(f"  Multipliers: min={profile['mult_min']:.1f}, max={profile['mult_max']:.1f}, mean={profile['mult_mean']:.1f}, median={profile['mult_median']:.1f}")
        print(f"  Style: {'EXTREME' if profile['extreme_comparisons'] > n * 0.2 else 'CONSERVATIVE' if profile['conservative_comparisons'] > n * 0.8 else 'BALANCED'}")
        print(f"  Log-ratio: mean={profile['log_ratio_mean']:.2f}, std={profile['log_ratio_std']:.2f}")
        print(f"  Category preferences (win rate):")
        for cat in sorted(profile['cat_win_rates'].keys(), key=lambda x: -profile['cat_win_rates'][x]):
            rate = profile['cat_win_rates'][cat]
            wins = cat_wins[cat]
            losses = cat_losses[cat]
            if wins + losses >= 2:  # Only show if multiple occurrences
                bias = "+++" if rate > 0.7 else "++" if rate > 0.6 else "+" if rate > 0.5 else "-" if rate < 0.4 else "--" if rate < 0.3 else ""
                print(f"    {cat:6s}: {rate:.2f} ({wins}W/{losses}L) {bias}")

# Identify juror archetypes
print("\n" + "="*100)
print("JUROR ARCHETYPES")
print("="*100)

# Cluster by multiplier style
extreme_jurors = [j for j, p in juror_profiles.items() if p['extreme_comparisons'] > p['n_comparisons'] * 0.2 and p['n_comparisons'] >= 10]
conservative_jurors = [j for j, p in juror_profiles.items() if p['conservative_comparisons'] > p['n_comparisons'] * 0.8 and p['n_comparisons'] >= 10]
balanced_jurors = [j for j in juror_profiles.keys() if j not in extreme_jurors and j not in conservative_jurors and juror_profiles[j]['n_comparisons'] >= 10]

print(f"\nExtreme Multiplier Users (100+x frequently): {len(extreme_jurors)}")
print(f"  {', '.join(extreme_jurors)}")

print(f"\nConservative Multiplier Users (mostly <10x): {len(conservative_jurors)}")
print(f"  {', '.join(conservative_jurors)}")

print(f"\nBalanced Multiplier Users: {len(balanced_jurors)}")
print(f"  {', '.join(balanced_jurors)}")

# Identify category specialists
print("\n" + "="*100)
print("CATEGORY BIAS ANALYSIS")
print("="*100)

category_biases = defaultdict(list)
for juror_id, profile in juror_profiles.items():
    if profile['n_comparisons'] >= 10:
        for cat, rate in profile['cat_win_rates'].items():
            wins = Counter(c['winner_cat'] for c in jurors[juror_id])[cat]
            losses = Counter(c['loser_cat'] for c in jurors[juror_id])[cat]
            if wins + losses >= 3:  # At least 3 occurrences
                category_biases[cat].append((juror_id, rate, wins, losses))

for cat in ['EL', 'CL', 'TOOLS', 'LANG', 'SPECS', 'INFRA']:
    if cat in category_biases:
        print(f"\n{cat} Category:")
        biased = sorted(category_biases[cat], key=lambda x: -x[1])
        for juror, rate, wins, losses in biased[:5]:
            print(f"  {juror}: {rate:.2f} win rate ({wins}W/{losses}L)")

# Key insights summary
print("\n" + "="*100)
print("KEY INSIGHTS FOR MODEL DESIGN")
print("="*100)

print("\n1. MULTIPLIER HETEROGENEITY:")
print(f"   - Extreme users: {len(extreme_jurors)} jurors (use 100+x frequently)")
print(f"   - Conservative users: {len(conservative_jurors)} jurors (rarely exceed 10x)")
print(f"   - This suggests per-juror scale factors are essential")

print("\n2. CATEGORY PREFERENCES:")
all_cat_rates = []
for profile in juror_profiles.values():
    if profile['n_comparisons'] >= 10:
        all_cat_rates.extend(profile['cat_win_rates'].values())
if all_cat_rates:
    print(f"   - Category win rates vary from {min(all_cat_rates):.2f} to {max(all_cat_rates):.2f}")
    print(f"   - Mean: {sum(all_cat_rates)/len(all_cat_rates):.2f}, Std: {(sum((x-sum(all_cat_rates)/len(all_cat_rates))**2 for x in all_cat_rates)/len(all_cat_rates))**0.5:.2f}")
    print(f"   - This suggests per-juror category effects are needed")

print("\n3. JURORS WITH ENOUGH DATA FOR INDIVIDUAL MODELING:")
rich_jurors = [j for j, p in juror_profiles.items() if p['n_comparisons'] >= 20]
print(f"   - {len(rich_jurors)} jurors with 20+ comparisons")
print(f"   - These jurors account for {sum(juror_profiles[j]['n_comparisons'] for j in rich_jurors)} / {len(samples)} comparisons ({100*sum(juror_profiles[j]['n_comparisons'] for j in rich_jurors)/len(samples):.1f}%)")

print("\n4. RECOMMENDATION:")
print("   - Implement mixed-effects model with:")
print("     * Per-juror category bias (6 parameters per juror)")
print("     * Per-juror scale factor (1 parameter per juror)")
print("     * Hierarchical priors (pool similar jurors)")
print("   - Expected parameter count: 7 × 40 jurors = 280 extra parameters")
print("   - With regularization, this is manageable")

# Save juror profiles for later use
with open("juror_profiles.json", "w") as f:
    # Convert to serializable format
    serializable = {}
    for jid, prof in juror_profiles.items():
        serializable[jid] = {
            'n_comparisons': prof['n_comparisons'],
            'mult_stats': {
                'min': prof['mult_min'],
                'max': prof['mult_max'],
                'mean': prof['mult_mean'],
                'median': prof['mult_median'],
            },
            'log_ratio_stats': {
                'mean': prof['log_ratio_mean'],
                'std': prof['log_ratio_std'],
            },
            'cat_win_rates': prof['cat_win_rates'],
            'style': 'extreme' if jid in extreme_jurors else 'conservative' if jid in conservative_jurors else 'balanced',
        }
    json.dump(serializable, f, indent=2)
    print("\n✓ Saved juror profiles to juror_profiles.json")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
