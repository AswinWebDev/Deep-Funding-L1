#!/usr/bin/env python3
"""
Deep Fund L1 competition model (seed repos only)

Approach
- Solve the exact organizer cost: least-squares on pairwise comparisons in log-space.
- Include non-seed repos as latent variables to better anchor seed repos (default). Use --exclude_nonseeds to disable.
- Add weak, domain-informed priors via ridge regularization toward z0 (log-weights prior).
- Weight samples by information content (|log-multiplier|^p) and balance by juror.
- Tune lambda and weight power p with juror-level cross-validation.
- Output weights for the 45 seed repos (sum to 1) in the exact order from dataset/test.csv.

Usage
  python script.py \
    --train dataset/train.csv \
    --test dataset/test.csv \
    --seeds seedRepos.json

It writes a timestamped submission file: submission_YYYYMMDD_HHMMSS.csv in the working directory
(by default). Use --outdir to change the destination directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Iterable, Set

import numpy as np
try:
    import requests  # optional; only used if --fetch_metrics passed
except Exception:  # pragma: no cover
    requests = None


# -----------------------------
# Utility parsing helpers
# -----------------------------

# URL migrations for repos that changed organizations
URL_MIGRATIONS = {
    "argotorg/solidity": "ethereum/solidity",
    "argotorg/sourcify": "ethereum/sourcify",
    "argotorg/fe": "ethereum/fe",
    "argotorg/act": "ethereum/act",
    "argotorg/hevm": "ethereum/hevm",
}

def repo_slug(url: str) -> str:
    """Return the canonical owner/repo slug from a GitHub URL string.
    Ex: https://github.com/ethereum/go-ethereum -> ethereum/go-ethereum
    Applies known URL migrations (e.g., argotorg -> ethereum).
    """
    if not isinstance(url, str):
        return str(url)
    parts = url.split("github.com/")
    if len(parts) < 2:
        slug = url.strip("/")
    else:
        tail = parts[1]
        # drop trailing slash or extra path components beyond owner/repo
        comps = [c for c in tail.split("/") if c]
        if len(comps) >= 2:
            slug = comps[0] + "/" + comps[1]
        else:
            slug = tail.strip("/")
    
    # Apply known migrations (CRITICAL FIX for URL aliasing bug)
    slug = URL_MIGRATIONS.get(slug, slug)
    return slug


def read_seeds(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(x).strip() for x in data]


@dataclass
class Sample:
    juror: str
    a: int  # index of repo_a in repo index
    b: int  # index of repo_b in repo index
    c: float  # log multiplier: ln(r_B) - ln(r_A)


# -----------------------------
# Categorization and priors
# -----------------------------

EL_ALIASES = {
    "ethereum/go-ethereum": 43,
    "nethermindeth/nethermind": 36,
    "hyperledger/besu": 16,
    "erigontech/erigon": 3,
    "paradigmxyz/reth": 2,
    "erigontech/silkworm": 1,
}

CL_ALIASES = {
    "sigp/lighthouse": 36,
    "prysmaticlabs/prysm": 33,
    "consensys/teku": 16,
    "status-im/nimbus-eth2": 10,
    "chainsafe/lodestar": 2,
    "grandinetech/grandine": 1,
    "lambdaclass/lambda_ethereum_consensus": 1,
}

# heuristics for other categories
TOOLS_HINT = {
    "ethers-io/ethers.js",
    "wevm/viem",
    "ethereum/web3.py",
    "hyperledger-web3j/web3j",
    "nethereum/nethereum",
    "nomicfoundation/hardhat",
    "foundry-rs/foundry",
    "ethereum/remix-project",
    "scaffold-eth/scaffold-eth-2",
}

LANG_HINT = {
    "argotorg/solidity",
    "vyperlang/vyper",
    "argotorg/fe",
    "vyperlang/titanoboa",
    "ethereum/py-evm",
    "ethereum/evmone",
    "argotorg/hevm",
    "argotorg/act",
    "ethdebug/format",
}

SPECS_HINT = {
    "ethereum/eips",
    "ethereum/consensus-specs",
    "ethereum/execution-apis",
    "ethereum-lists/chains",
}

INFRA_HINT = {
    "ethpandaops/ethereum-package",
    "ethpandaops/ethereum-helm-charts",
    "ethpandaops/checkpointz",
}

CATEGORY_SHARES = {
    # Updated based on NEW juror patterns (Sept 2025 analysis)
    # NEW data shows: SPECS 8.5% (was 1.8%), foundational libs emphasized
    "EL": 0.38,      # Was 0.46 (-8%: less emphasis on client diversity alone)
    "CL": 0.22,      # Was 0.27 (-5%: same reasoning)
    "TOOLS": 0.20,   # Was 0.13 (+7%: foundational libraries like ethers.js, hardhat)
    "LANG": 0.08,    # Was 0.09 (-1%: solidity overweighted, 0.0 dep rank)
    "SPECS": 0.08,   # Was 0.03 (+5%: NEW jurors care about protocol specs!)
    "INFRA": 0.04,   # Was 0.02 (+2%: ethereum-package, helm-charts)
}


def categorize(slug: str) -> str:
    if slug in EL_ALIASES:
        return "EL"
    if slug in CL_ALIASES:
        return "CL"
    if slug in TOOLS_HINT:
        return "TOOLS"
    if slug in LANG_HINT:
        return "LANG"
    if slug in SPECS_HINT:
        return "SPECS"
    if slug in INFRA_HINT:
        return "INFRA"
    # Fallbacks by substring heuristics
    s = slug.lower()
    if any(k in s for k in ["geth", "nethermind", "besu", "erigon", "reth", "silkworm"]):
        return "EL"
    if any(k in s for k in ["lighthouse", "prysm", "teku", "nimbus", "lodestar", "grandine", "consensus"]):
        return "CL"
    if any(k in s for k in ["solidity", "vyper", "titanoboa", "fe", "py-evm", "evmone", "hevm", "act", "format"]):
        return "LANG"
    if any(k in s for k in ["eips", "consensus-specs", "execution-apis", "chains"]):
        return "SPECS"
    if any(k in s for k in ["pandaops", "helm", "checkpoint"]):
        return "INFRA"
    return "TOOLS"


def build_priors(repo_slugs: List[str], seed_set: Set[str]) -> np.ndarray:
    """Construct a weak prior z0 over all repos.
    - Allocate category-level mass across seeds according to CATEGORY_SHARES.
    - Within EL/CL use known market shares where available; otherwise uniform within category.
    - BOOST by OSO dependency rank (objective measure of foundational importance).
    - BOOST by dependency graph counts (how many seed repos depend on this).
    - Non-seed repos get a tiny fraction of the mass of their category (helps identifiability).
    Returns z0 (log-weights), normalized so exp(z0) sums to 1 across all repos considered.
    """
    # Split seeds by category
    seeds_by_cat: Dict[str, List[str]] = defaultdict(list)
    for s in seed_set:
        seeds_by_cat[categorize(s)].append(s)

    # Numerical mass per seed based on category and optionally market share
    prior_mass: Dict[str, float] = {slug: 0.0 for slug in repo_slugs}

    # EL category with market shares
    if seeds_by_cat.get("EL"):
        total_share = sum(EL_ALIASES.get(s, 1.0) for s in seeds_by_cat["EL"]) or 1.0
        for s in seeds_by_cat["EL"]:
            share = EL_ALIASES.get(s, 1.0) / total_share
            prior_mass[s] = CATEGORY_SHARES["EL"] * share

    # CL category with market shares
    if seeds_by_cat.get("CL"):
        total_share = sum(CL_ALIASES.get(s, 1.0) for s in seeds_by_cat["CL"]) or 1.0
        for s in seeds_by_cat["CL"]:
            share = CL_ALIASES.get(s, 1.0) / total_share
            prior_mass[s] = CATEGORY_SHARES["CL"] * share

    # Other categories: uniform within category
    for cat in ["TOOLS", "LANG", "SPECS", "INFRA"]:
        lst = seeds_by_cat.get(cat, [])
        if lst:
            for s in lst:
                prior_mass[s] = CATEGORY_SHARES[cat] / len(lst)

    # Non-seeds get tiny mass (so they exist but don't eat the pie)
    non_seeds = [r for r in repo_slugs if r not in seed_set]
    tiny = 1e-6
    for r in non_seeds:
        prior_mass[r] = tiny

    # ===== PRIORITY FIX #1: OSO Dependency Rank Boost =====
    # Load OSO metadata (official measure of foundational library importance)
    oso_path = "dependency-graph-main/datasets/pairwise/getProjectMetadata.json"
    oso_data: Dict[str, Dict] = {}
    
    if os.path.exists(oso_path):
        try:
            with open(oso_path, "r", encoding="utf-8") as f:
                oso_raw = json.load(f)
            
            print(f"[OSO] Loaded {len(oso_raw)} project entries from metadata")
            
            # Normalize to slug keys (repo_slug now handles URL migrations)
            for url, metrics in oso_raw.items():
                slug = repo_slug(url)
                oso_data[slug] = metrics
            
            # Verify seed coverage
            seeds_with_oso = [s for s in seed_set if s in oso_data]
            seeds_missing = [s for s in seed_set if s not in oso_data]
            
            print(f"[OSO] Coverage: {len(seeds_with_oso)}/45 seeds have OSO data")
            if seeds_missing and len(seeds_missing) <= 10:
                print(f"[OSO] Missing seeds: {', '.join(s.split('/')[-1] for s in seeds_missing)}")
            elif seeds_missing:
                print(f"[OSO] WARNING: {len(seeds_missing)} seeds missing OSO data!")
                print(f"[OSO] First 5 missing: {', '.join(s.split('/')[-1] for s in seeds_missing[:5])}")
                
        except Exception as e:
            print(f"Warning: Could not load OSO metadata: {e}")
    
    # ===== PRIORITY FIX #2: Dependency Graph Count Boost =====
    # Load dependency graph (counts how many seed repos depend on each repo)
    dep_graph_path = "dependency-graph-main/datasets/v2-graph/dependency-graph-v2.csv"
    dep_counts: Dict[str, int] = defaultdict(int)
    if os.path.exists(dep_graph_path):
        try:
            with open(dep_graph_path, "r", encoding="utf-8") as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    dep_repo = repo_slug(row["dependency_repo"])
                    dep_counts[dep_repo] += 1
        except Exception as e:
            print(f"Warning: Could not load dependency graph: {e}")
    
    # Apply boosts to prior mass (ONLY to seeds, to preserve category balance)
    boost_log = []
    for slug in seed_set:
        base_mass = prior_mass[slug]
        total_boost = 1.0
        boost_details = {}
        
        # OSO Dependency Rank boost (0.0 to ~0.87 for ethers.js)
        dep_rank = oso_data.get(slug, {}).get("osoDependencyRank", 0.0)
        if dep_rank > 0.0:
            # Conservative: 2x for 0.8+ rank
            # ethers.js (0.869) -> 2.7x, OZ (0.698) -> 2.4x, hardhat (0.595) -> 2.2x
            boost_rank = 1.0 + 2.0 * dep_rank
            total_boost *= boost_rank
            boost_details['depRank'] = (dep_rank, boost_rank)
        
        # NEW: Funding boost (community validation signal)
        funding_usd = oso_data.get(slug, {}).get("totalFundingUsd", 0.0)
        if funding_usd > 100000:  # $100k threshold
            # Conservative log-scale: 1.5x at $1M, 2.0x at $2.6M (geth)
            boost_funding = 1.0 + 0.5 * math.log10(funding_usd / 100000)
            total_boost *= boost_funding
            boost_details['funding'] = (funding_usd / 1e6, boost_funding)
        
        # NEW: Developer commitment boost
        avg_devs = oso_data.get(slug, {}).get("avgFullTimeDevs", 0.0)
        if avg_devs >= 1.0:
            # Conservative linear: 1.5x at 5 devs, 2.0x at 10 devs, cap at 2.5x
            boost_devs = 1.0 + 0.1 * min(avg_devs, 15.0)
            total_boost *= boost_devs
            boost_details['devs'] = (avg_devs, boost_devs)
        
        # Dependency graph count boost (0 to 89 for alloy)
        num_deps = dep_counts.get(slug, 0)
        if num_deps > 0:
            # Conservative: 0.02 per dependency
            # alloy (89) -> 2.78x, ethers.js (28) -> 1.56x, hardhat (19) -> 1.38x
            boost_deps = 1.0 + 0.02 * num_deps
            total_boost *= boost_deps
            boost_details['deps'] = (num_deps, boost_deps)
        
        if total_boost > 1.01:  # Log if boosted significantly
            prior_mass[slug] *= total_boost
            boost_log.append((slug, total_boost, boost_details))
    
    # Enhanced logging with detailed boost breakdown
    if boost_log:
        boost_log.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[priors] OSO boosts applied to {len(boost_log)} seeds (top 15):")
        print(f"{'Repository':<30} {'Total':>7} {'Details':<60}")
        print("-" * 100)
        for slug, total, details in boost_log[:15]:
            repo_name = slug.split('/')[-1][:28]
            detail_parts = []
            if 'depRank' in details:
                detail_parts.append(f"depRank={details['depRank'][0]:.3f}->{details['depRank'][1]:.2f}x")
            if 'funding' in details:
                detail_parts.append(f"fund=${details['funding'][0]:.1f}M->{details['funding'][1]:.2f}x")
            if 'devs' in details:
                detail_parts.append(f"devs={details['devs'][0]:.1f}->{details['devs'][1]:.2f}x")
            if 'deps' in details:
                detail_parts.append(f"deps={details['deps'][0]}->{details['deps'][1]:.2f}x")
            detail_str = ", ".join(detail_parts)[:58]
            print(f"{repo_name:<30} {total:>6.2f}x {detail_str:<60}")
    
    # Convert to vector aligned to repo_slugs
    v = np.array([prior_mass.get(s, tiny) for s in repo_slugs], dtype=float)
    v = np.clip(v, 1e-12, None)
    v = v / v.sum()  # ensure sums to 1 across all repos
    z0 = np.log(v)
    return z0


# -----------------------------
# Feature engineering (optional metrics-informed prior)
# -----------------------------

def load_external_metrics(path: str | None) -> Dict[str, Dict[str, float]]:
    """Load metrics JSON mapping 'owner/repo' -> dict of metrics.
    Expected keys (all optional): stars, forks, watchers, open_issues,
    contributors, created_at_unix, pushed_at_unix, commits_1y, releases_1y.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize keys to slugs
    out: Dict[str, Dict[str, float]] = {}
    for k, v in data.items():
        out[repo_slug(k)] = v
    return out


def maybe_fetch_github_metrics(repos: List[str], out_path: str, token_env: str = "GITHUB_TOKEN") -> None:
    """Fetch lightweight GitHub repo metrics and store to JSON.
    Respects rate limits; provide a GitHub token in env GITHUB_TOKEN for higher limits.
    Safe to call even if 'requests' is not installed; it will no-op.
    """
    if requests is None:
        print("requests not available; skip fetching metrics")
        return
    token = os.environ.get(token_env)
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = "https://api.github.com/repos/"
    out: Dict[str, Dict[str, float]] = {}
    for slug in repos:
        url = base + slug
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"warn: failed {slug} {r.status_code}")
            continue
        j = r.json()
        m: Dict[str, float] = {
            "stars": float(j.get("stargazers_count", 0) or 0),
            "forks": float(j.get("forks_count", 0) or 0),
            "watchers": float(j.get("subscribers_count", 0) or 0),
            "open_issues": float(j.get("open_issues_count", 0) or 0),
        }
        # timestamps
        def _parse_dt(s: str | None) -> float:
            if not s:
                return 0.0
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0
        m["created_at_unix"] = _parse_dt(j.get("created_at"))
        m["pushed_at_unix"] = _parse_dt(j.get("pushed_at"))
        out[slug] = m
        time.sleep(0.2)  # be polite
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def compute_diversity_feature(slug: str) -> float:
    """Approximate marginal contribution to client diversity via Herfindahl delta.
    Positive if removing the repo increases concentration.
    """
    if slug in EL_ALIASES:
        shares = [v for v in EL_ALIASES.values()]
        s = EL_ALIASES[slug]
    elif slug in CL_ALIASES:
        shares = [v for v in CL_ALIASES.values()]
        s = CL_ALIASES[slug]
    else:
        return 0.0
    shares_all = np.array(shares, dtype=float)
    H_with = float(np.sum(shares_all ** 2) / (np.sum(shares_all) ** 2))
    shares_wo = shares_all[shares_all != s]
    if shares_wo.size == 0:
        return 0.0
    H_wo = float(np.sum(shares_wo ** 2) / (np.sum(shares_wo) ** 2))
    # Removal makes H increase (more concentrated): signal of importance
    return max(H_wo - H_with, 0.0)


def build_feature_matrix(repo_slugs: List[str], metrics: Dict[str, Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """Return (X, feature_names), with rows aligned to repo_slugs.
    Features include:
    - Category one-hots (EL/CL/TOOLS/LANG/SPECS/INFRA)
    - Market share within EL/CL (normalized)
    - Diversity contribution (Herfindahl delta)
    - GitHub metrics (log1p): stars, forks, watchers, open_issues
    - Recency: years since created, years since last push (clamped)
    - Extended signals (learned, no hardcoding of outputs) when available in metrics:
      topics_count, has_topic_ethereum, languages_hhi, language shares,
      contributors, releases_count, latest_release recency, per-year rates for
      stars/forks/watchers/releases, issue backlog per release, topic density,
      recent activity flags (push/release in last 6 months)
    """
    cats = ["EL", "CL", "TOOLS", "LANG", "SPECS", "INFRA"]
    feat_names: List[str] = [f"cat_{c}" for c in cats] + ["el_share", "cl_share", "diversity"]
    gh_names = ["stars", "forks", "watchers", "open_issues", "created_at_unix", "pushed_at_unix"]
    feat_names += [f"gh_{k}" for k in gh_names]
    # Extended feature names (append-only order for stability)
    feat_names += [
        "gh_topics_count", "gh_has_topic_ethereum",
        "lang_hhi",
        "lang_solidity", "lang_go", "lang_rust", "lang_typescript", "lang_javascript", "lang_python", "lang_cpp",
        "contributors_log1p", "releases_count_log1p",
        "yrs_since_last_release", "releases_per_year_log1p",
        "rate_stars_per_year_log1p", "rate_forks_per_year_log1p", "rate_watchers_per_year_log1p",
        "issue_backlog_per_release_log1p", "topics_density",
        "recent_push_6m", "recent_release_6m",
    ]

    X = np.zeros((len(repo_slugs), len(feat_names)), dtype=float)
    # Precompute totals
    el_total = sum(EL_ALIASES.values()) or 1.0
    cl_total = sum(CL_ALIASES.values()) or 1.0
    now_ts = time.time()
    sec_per_year = 365.25 * 24 * 3600

    for i, s in enumerate(repo_slugs):
        cat = categorize(s)
        # category one-hots
        for j, c in enumerate(cats):
            if cat == c:
                X[i, j] = 1.0
        # market shares
        X[i, len(cats) + 0] = (EL_ALIASES.get(s, 0.0) / el_total) if cat == "EL" else 0.0
        X[i, len(cats) + 1] = (CL_ALIASES.get(s, 0.0) / cl_total) if cat == "CL" else 0.0
        # diversity contribution
        X[i, len(cats) + 2] = compute_diversity_feature(s)
        # GitHub metrics
        m = metrics.get(s, {})
        stars = float(m.get("stars", 0.0))
        forks = float(m.get("forks", 0.0))
        watchers = float(m.get("watchers", 0.0))
        open_issues = float(m.get("open_issues", 0.0))
        created = float(m.get("created_at_unix", 0.0))
        pushed = float(m.get("pushed_at_unix", 0.0))
        X[i, len(cats) + 3 + 0] = math.log1p(stars)
        X[i, len(cats) + 3 + 1] = math.log1p(forks)
        X[i, len(cats) + 3 + 2] = math.log1p(watchers)
        X[i, len(cats) + 3 + 3] = math.log1p(open_issues)
        # recency proxies (years)
        yrs_created = (now_ts - created) / sec_per_year if created > 0 else 5.0
        yrs_pushed = (now_ts - pushed) / sec_per_year if pushed > 0 else 1.0
        X[i, len(cats) + 3 + 4] = max(min(yrs_created, 15.0), 0.0)
        X[i, len(cats) + 3 + 5] = max(min(yrs_pushed, 15.0), 0.0)
        # Extended metrics with safe defaults
        topics_count = float(m.get("topics_count", 0.0))
        has_topic_eth = float(m.get("has_topic_ethereum", 0.0))
        lang_hhi = float(m.get("languages_hhi", 0.0))
        # Per-language shares if provided
        lang_sol = float(m.get("lang_Solidity_share", 0.0))
        lang_go = float(m.get("lang_Go_share", 0.0))
        lang_rs = float(m.get("lang_Rust_share", 0.0))
        lang_ts = float(m.get("lang_TypeScript_share", 0.0))
        lang_js = float(m.get("lang_JavaScript_share", 0.0))
        lang_py = float(m.get("lang_Python_share", 0.0))
        lang_cpp = float(m.get("lang_C++_share", 0.0))
        contributors = float(m.get("contributors", 0.0))
        releases_count = float(m.get("releases_count", 0.0))
        latest_release = float(m.get("latest_release_unix", 0.0))
        yrs_last_release = (now_ts - latest_release) / sec_per_year if latest_release > 0 else 5.0
        years = max(yrs_created, 0.25)
        # Derived per-year rates
        r_stars_py = stars / years
        r_forks_py = forks / years
        r_watchers_py = watchers / years
        r_releases_py = releases_count / years
        issue_backlog_per_release = open_issues / (releases_count + 1.0)
        topics_density = topics_count / max(math.log1p(stars + forks + watchers), 1.0)
        recent_push_6m = 1.0 if yrs_pushed <= 0.5 else 0.0
        recent_release_6m = 1.0 if yrs_last_release <= 0.5 else 0.0
        # Write extended block
        base = len(cats) + 3 + 6
        X[i, base + 0] = math.log1p(topics_count)
        X[i, base + 1] = has_topic_eth
        X[i, base + 2] = lang_hhi
        X[i, base + 3] = lang_sol
        X[i, base + 4] = lang_go
        X[i, base + 5] = lang_rs
        X[i, base + 6] = lang_ts
        X[i, base + 7] = lang_js
        X[i, base + 8] = lang_py
        X[i, base + 9] = lang_cpp
        X[i, base + 10] = math.log1p(contributors)
        X[i, base + 11] = math.log1p(releases_count)
        X[i, base + 12] = max(min(yrs_last_release, 15.0), 0.0)
        X[i, base + 13] = math.log1p(r_releases_py)
        X[i, base + 14] = math.log1p(r_stars_py)
        X[i, base + 15] = math.log1p(r_forks_py)
        X[i, base + 16] = math.log1p(r_watchers_py)
        X[i, base + 17] = math.log1p(issue_backlog_per_release)
        X[i, base + 18] = float(topics_density)
        X[i, base + 19] = recent_push_6m
        X[i, base + 20] = recent_release_6m
    return X, feat_names


def fit_theta_from_pairs(X: np.ndarray, samples: List[Sample], weights_by_sample: List[float], lam_theta: float) -> np.ndarray:
    """Ridge regression for c ≈ (x_b - x_a)·theta.
    Returns theta.
    """
    d_rows: List[np.ndarray] = []
    y: List[float] = []
    W: List[float] = []
    for s, w in zip(samples, weights_by_sample):
        d = X[s.b] - X[s.a]
        d_rows.append(d)
        y.append(s.c)
        W.append(w)
    D = np.vstack(d_rows)
    yv = np.array(y)
    Wv = np.array(W)
    # Solve (D^T W D + lam I) theta = D^T W y
    # Scale columns for numerical stability
    col_scale = np.sqrt(np.maximum((D ** 2 * Wv[:, None]).sum(axis=0), 1e-12))
    Dn = D / col_scale
    A = (Dn.T * Wv) @ Dn + lam_theta * np.eye(Dn.shape[1])
    rhs = (Dn.T * Wv) @ yv
    try:
        theta_n = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        theta_n, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    theta = theta_n / col_scale
    return theta


# -----------------------------
# Data ingestion
# -----------------------------

def read_train_samples(train_path: str, repo_to_idx: Dict[str, int]) -> Tuple[List[Sample], Dict[str, int]]:
    samples: List[Sample] = []
    juror_counts: Dict[str, int] = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            juror = row.get("juror", "?").strip()
            a_slug = repo_slug(row["repo_a"])  # type: ignore
            b_slug = repo_slug(row["repo_b"])  # type: ignore
            if a_slug not in repo_to_idx or b_slug not in repo_to_idx:
                # unseen repo from indexing stage? skip (shouldn't happen)
                continue
            a = repo_to_idx[a_slug]
            b = repo_to_idx[b_slug]
            choice = int(str(row["choice"]).strip())
            multiplier = float(str(row["multiplier"]).strip())
            multiplier = max(multiplier, 1e-9)
            # c = ln(r_B) - ln(r_A)
            c = math.log(multiplier)
            if choice == 1:
                c = -c
            elif choice == 2:
                c = +c
            else:
                # invalid choice; skip
                continue
            samples.append(Sample(juror=juror, a=a, b=b, c=c))
            juror_counts[juror] += 1
    return samples, juror_counts


def index_repos(train_path: str, seed_urls: List[str], include_nonseeds: bool = True) -> Tuple[List[str], Dict[str, int]]:
    seed_slugs = [repo_slug(u) for u in seed_urls]
    repo_set: Set[str] = set(seed_slugs)
    if include_nonseeds:
        with open(train_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                repo_a = repo_slug(row["repo_a"])  # type: ignore
                repo_b = repo_slug(row["repo_b"])  # type: ignore
                repo_set.add(repo_a)
                repo_set.add(repo_b)
    repo_slugs = sorted(repo_set)
    repo_to_idx = {s: i for i, s in enumerate(repo_slugs)}
    return repo_slugs, repo_to_idx


# -----------------------------
# Linear system construction and solver
# -----------------------------

def build_normal_equations(
    n: int,
    samples: Iterable[Sample],
    juror_counts: Dict[str, int],
    weight_power: float = 0.0,
    balance_by_juror: bool = False,
    repo_slugs: List[str] | None = None,
    seed_set: Set[str] | None = None,
    seeds_boost: float = 1.0,
    cap_logmult: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct A (n x n), b (n), and degree array (n) from samples.
    For weighted LS minimizing sum w (x_i^T z - y_i)^2 where x_i has -1 at a and +1 at b, y_i=c.
    Returns A, b, degree where degree[j] is number of comparisons involving repo j.
    """
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    degree = np.zeros(n, dtype=float)

    for s in samples:
        # sample weight options
        # default (eval-aligned): uniform weights
        w = abs(s.c) ** weight_power if weight_power > 0 else 1.0
        if balance_by_juror:
            jc = max(1, juror_counts.get(s.juror, 1))
            w /= jc
        w = max(w, 1e-9)
        # Boost seeds-only pairs if configured
        if seeds_boost != 1.0 and repo_slugs is not None and seed_set is not None:
            try:
                sa = repo_slugs[s.a]
                sb = repo_slugs[s.b]
                if sa in seed_set and sb in seed_set:
                    w *= seeds_boost
            except Exception:
                pass
        # y target with optional capping
        y = s.c
        if cap_logmult is not None:
            if y > cap_logmult:
                y = cap_logmult
            elif y < -cap_logmult:
                y = -cap_logmult
        # Update normal equations
        A[s.a, s.a] += w
        A[s.b, s.b] += w
        A[s.a, s.b] += -w
        A[s.b, s.a] += -w
        b[s.a] += -w * y
        b[s.b] += +w * y
        degree[s.a] += 1
        degree[s.b] += 1
    # Add a tiny diagonal for numerical stability
    A.flat[:: n + 1] += 1e-9
    return A, b, degree


def solve_ridge(
    A: np.ndarray,
    b: np.ndarray,
    z0: np.ndarray,
    degree: np.ndarray,
    lam: float = 0.3,
) -> np.ndarray:
    """Solve (A + lam * Gamma^T Gamma) z = b + lam * Gamma^T Gamma z0
    with Gamma diagonal ~ 1 / sqrt(1 + degree) so less-anchored for well-constrained repos.
    """
    n = A.shape[0]
    gamma = 1.0 / np.sqrt(1.0 + degree)
    reg_diag = lam * (gamma ** 2)
    A_reg = A.copy()
    A_reg.flat[:: n + 1] += reg_diag  # add to diagonal
    rhs = b + reg_diag * z0
    # Solve
    try:
        z = np.linalg.solve(A_reg, rhs)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        z, *_ = np.linalg.lstsq(A_reg, rhs, rcond=None)
    return z


def evaluate_cost(z: np.ndarray, samples: Iterable[Sample]) -> float:
    return sum((z[s.b] - z[s.a] - s.c) ** 2 for s in samples)


def evaluate_mse(z: np.ndarray, samples: Iterable[Sample]) -> float:
    samples_list = list(samples)
    if not samples_list:
        return 0.0
    return evaluate_cost(z, samples_list) / float(len(samples_list))


def build_normal_equations_custom(
    n: int,
    samples: List[Sample],
    weights: List[float],
    y_targets: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Custom normal equations from arbitrary per-sample weights and targets.
    x has -1 at a and +1 at b; target y is the effective RHS.
    """
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    degree = np.zeros(n, dtype=float)
    for s, w, y in zip(samples, weights, y_targets):
        w = max(float(w), 1e-12)
        A[s.a, s.a] += w
        A[s.b, s.b] += w
        A[s.a, s.b] += -w
        A[s.b, s.a] += -w
        b[s.a] += -w * y
        b[s.b] += +w * y
        degree[s.a] += 1
        degree[s.b] += 1
    A.flat[:: n + 1] += 1e-9
    return A, b, degree


def solve_with_juror_scale_and_huber(
    n: int,
    samples: List[Sample],
    juror_counts: Dict[str, int],
    z_prior: np.ndarray,
    lam_z: float,
    repo_slugs: List[str],
    seed_set: Set[str],
    weight_power: float,
    *,
    balance_by_juror: bool,
    seeds_boost: float,
    cap_logmult: float | None,
    use_huber: bool,
    huber_delta: float,
    use_juror_scale: bool,
    lam_t: float,
    t_clip: Tuple[float, float],
    verbose: bool,
    # Juror reliability reweighting
    use_reliability: bool = False,
    rel_alpha: float = 1.0,
    rel_min: float = 0.3,
    reliability_seeds_only: bool = True,
    # Per-juror additive bias
    use_bias: bool = False,
    lam_b: float = 0.3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Alternating optimization for z and per-juror scales t_j with optional Huber IRLS.
    Returns (z, juror_scales).
    """
    # Base weights
    base_w: List[float] = []
    for s in samples:
        w = abs(s.c) ** weight_power if weight_power > 0 else 1.0
        if balance_by_juror:
            jc = max(1, juror_counts.get(s.juror, 1))
            w /= jc
        if seeds_boost != 1.0:
            sa = repo_slugs[s.a]
            sb = repo_slugs[s.b]
            if sa in seed_set and sb in seed_set:
                w *= seeds_boost
        base_w.append(max(float(w), 1e-9))
    # Targets with capping
    y_raw: List[float] = []
    for s in samples:
        y = float(s.c)
        if cap_logmult is not None:
            if y > cap_logmult:
                y = cap_logmult
            elif y < -cap_logmult:
                y = -cap_logmult
        y_raw.append(y)
    # Initialize
    z = z_prior.copy()
    t: Dict[str, float] = {j: 1.0 for j in juror_counts.keys()}
    w_huber: List[float] = [1.0] * len(samples)
    # Juror reliability weights (per-juror inverse-variance style)
    rel_w_by_j: Dict[str, float] = {j: 1.0 for j in juror_counts.keys()}
    # Juror additive bias b_j (ridge to 0)
    b_bias: Dict[str, float] = {j: 0.0 for j in juror_counts.keys()}
    max_iters = 5
    for it in range(max_iters):
        # Build effective weights and targets using current t and huber
        eff_w: List[float] = []
        eff_y: List[float] = []
        for idx, s in enumerate(samples):
            tj = t.get(s.juror, 1.0)
            w = base_w[idx] * (tj * tj) * w_huber[idx]
            if use_reliability:
                w *= float(rel_w_by_j.get(s.juror, 1.0))
            eff_w.append(w)
            bj = b_bias.get(s.juror, 0.0)
            eff_y.append((y_raw[idx] - bj) / max(tj, 1e-6))
        A, b, degree = build_normal_equations_custom(n, samples, eff_w, eff_y)
        z = solve_ridge(A, b, z_prior, degree, lam=lam_z)
        # Residuals on original scale r = t_j*(z_b - z_a) + b_j - y_raw
        if use_juror_scale or use_bias:
            # Accumulate per-juror stats
            d_by_j: Dict[str, List[float]] = {}
            y_by_j: Dict[str, List[float]] = {}
            w_by_j: Dict[str, List[float]] = {}
            for idx, s in enumerate(samples):
                d = float(z[s.b] - z[s.a])
                d_by_j.setdefault(s.juror, []).append(d)
                y_by_j.setdefault(s.juror, []).append(y_raw[idx])
                w_by_j.setdefault(s.juror, []).append(base_w[idx] * w_huber[idx])
            if use_juror_scale:
                for j in d_by_j.keys():
                    ds = np.array(d_by_j[j], dtype=float)
                    ys = np.array(y_by_j[j], dtype=float)
                    ws = np.array(w_by_j[j], dtype=float)
                    ys_adj = ys - b_bias.get(j, 0.0)
                    denom = float(np.sum(ws * ds * ds) + lam_t)
                    num = float(np.sum(ws * ds * ys_adj) + lam_t * 1.0)
                    tj = num / max(denom, 1e-9)
                    tj = float(np.clip(tj, t_clip[0], t_clip[1]))
                    t[j] = tj
            if use_bias:
                for j in d_by_j.keys():
                    ds = np.array(d_by_j[j], dtype=float)
                    ys = np.array(y_by_j[j], dtype=float)
                    ws = np.array(w_by_j[j], dtype=float)
                    tj = t.get(j, 1.0)
                    # Solve b_j: argmin sum ws (tj*ds + b_j - ys)^2 + lam_b * b_j^2
                    # b_j = sum ws*(ys - tj*ds) / (sum ws + lam_b)
                    num = float(np.sum(ws * (ys - tj * ds)))
                    denom = float(np.sum(ws) + lam_b)
                    bj = num / max(denom, 1e-9)
                    b_bias[j] = float(bj)
        # Update Huber weights based on residuals with current t
        if use_huber:
            for i, s in enumerate(samples):
                r = float(t.get(s.juror, 1.0) * (z[s.b] - z[s.a]) + b_bias.get(s.juror, 0.0) - y_raw[i])
                ar = abs(r)
                w_huber[i] = 1.0 if ar <= huber_delta else (huber_delta / max(ar, 1e-9))
        # Update juror reliability weights using residual variance
        if use_reliability:
            resid_by_j: Dict[str, List[float]] = {}
            for i, s in enumerate(samples):
                # Optionally restrict to seeds-only pairs for reliability estimation
                if reliability_seeds_only:
                    sa = repo_slugs[s.a]
                    sb = repo_slugs[s.b]
                    if not (sa in seed_set and sb in seed_set):
                        continue
                r = float(t.get(s.juror, 1.0) * (z[s.b] - z[s.a]) + b_bias.get(s.juror, 0.0) - y_raw[i])
                resid_by_j.setdefault(s.juror, []).append(r)
            for j, rs in resid_by_j.items():
                if len(rs) == 0:
                    continue
                var = float(np.mean(np.square(np.array(rs, dtype=float))))
                wj = 1.0 / (1.0 + rel_alpha * var)
                if wj < rel_min:
                    wj = rel_min
                if wj > 1.0:
                    wj = 1.0
                rel_w_by_j[j] = wj
        if verbose:
            if use_juror_scale and it == max_iters - 1:
                vals = np.array(list(t.values()), dtype=float)
                if vals.size:
                    q = np.quantile(vals, [0.1, 0.5, 0.9])
                    print(f"[juror-scale] t_j quantiles p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}")
            if use_huber and it == max_iters - 1:
                inlier = sum(1 for w in w_huber if w >= 0.999)
                print(f"[huber] inliers={inlier}/{len(w_huber)} (delta={huber_delta})")
            if use_bias and it == max_iters - 1 and b_bias:
                bv = np.array(list(b_bias.values()), dtype=float)
                q = np.quantile(bv, [0.1, 0.5, 0.9])
                print(f"[juror-bias] b_j quantiles p10={q[0]:+.3f} p50={q[1]:+.3f} p90={q[2]:+.3f}")
            if use_reliability and it == max_iters - 1 and rel_w_by_j:
                rw = np.array(list(rel_w_by_j.values()), dtype=float)
                q = np.quantile(rw, [0.1, 0.5, 0.9])
                print(f"[reliability] juror weights p10={q[0]:.3f} p50={q[1]:.3f} p90={q[2]:.3f}")
    return z, t


def calibrate_category_offsets(
    repo_slugs: List[str],
    samples: List[Sample],
    seed_set: Set[str],
    z: np.ndarray,
    lam_cat: float = 0.3,
    seeds_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Solve for category-wise additive offsets alpha to minimize SSE on (optionally) seeds-only pairs.
    z' = z + alpha[cat(idx)]. Returns (z_calibrated, alpha, cat_names_ordered).
    """
    cat_names = ["EL", "CL", "TOOLS", "LANG", "SPECS", "INFRA"]
    cat_to_idx = {c: i for i, c in enumerate(cat_names)}
    # Build residual r and design X
    rows: List[List[float]] = []
    r: List[float] = []
    for s in samples:
        if seeds_only and not (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set):
            continue
        ca = cat_to_idx.get(categorize(repo_slugs[s.a]), None)
        cb = cat_to_idx.get(categorize(repo_slugs[s.b]), None)
        if ca is None or cb is None:
            continue
        row = [0.0] * len(cat_names)
        row[cb] = 1.0
        row[ca] = -1.0
        rows.append(row)
        r.append(s.c - (z[s.b] - z[s.a]))
    if not rows:
        return z, np.zeros(len(cat_names)), cat_names
    X = np.array(rows, dtype=float)
    rv = np.array(r, dtype=float)
    A = X.T @ X + lam_cat * np.eye(X.shape[1])
    b = X.T @ rv
    try:
        alpha = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Apply offsets
    z_cal = z.copy()
    for i, slug in enumerate(repo_slugs):
        cat = categorize(slug)
        z_cal[i] += alpha[cat_to_idx[cat]]
    return z_cal, alpha, cat_names


def calibrate_temperature(
    repo_slugs: List[str],
    samples: List[Sample],
    seed_set: Set[str],
    z: np.ndarray,
    seeds_only: bool = True,
) -> Tuple[np.ndarray, float]:
    """Fit a global temperature t to minimize sum (t*(z_b - z_a) - c)^2 on (optionally) seeds-only pairs.
    Returns (z_scaled, t)."""
    ds: List[float] = []
    cs: List[float] = []
    for s in samples:
        if seeds_only and not (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set):
            continue
        ds.append(float(z[s.b] - z[s.a]))
        cs.append(float(s.c))
    if not ds:
        return z, 1.0
    d = np.array(ds, dtype=float)
    c = np.array(cs, dtype=float)
    denom = float(np.sum(d * d))
    if denom <= 1e-12:
        return z, 1.0
    t = float(np.sum(d * c) / denom)
    z_scaled = z * t
    return z_scaled, t


# -----------------------------
# Juror-level cross-validation
# -----------------------------

def juror_kfold(jurors: List[str], k: int = 5, seed: int = 42) -> List[Set[str]]:
    rng = random.Random(seed)
    js = jurors[:]
    rng.shuffle(js)
    folds = [set() for _ in range(k)]
    for i, j in enumerate(js):
        folds[i % k].add(j)
    return folds

def counts_by_juror(samples: Iterable[Sample]) -> Dict[str, int]:
    c: Dict[str, int] = Counter()
    for s in samples:
        c[s.juror] = c.get(s.juror, 0) + 1
    return dict(c)


def fit_with_cv(
    samples: List[Sample],
    repo_slugs: List[str],
    seed_set: Set[str],
    lam_grid: List[float],
    p_grid: List[float],
    kfold: int = 5,
    rng_seed: int = 42,
    seeds_only_eval: bool = True,
    *,
    seeds_boost_grid: List[float] | None = None,
    tune_robust: bool = False,
    huber_delta_grid: List[float] | None = None,
    lam_t_grid: List[float] | None = None,
    t_clip: Tuple[float, float] = (0.8, 1.25),
    balance_by_juror: bool = False,
    cap_logmult: float | None = None,
    # New CV toggles
    tune_balance_by_juror: bool = False,
    cap_logmult_grid: List[float] | None = None,
    # Reliability reweighting
    tune_reliability: bool = False,
    rel_alpha_grid: List[float] | None = None,
    rel_min: float = 0.3,
    reliability_seeds_only: bool = True,
) -> Tuple[float, float, float, bool, float, bool, float]:
    jurors = sorted({s.juror for s in samples})
    folds = juror_kfold(jurors, kfold, rng_seed)
    seeds_boost_opts = seeds_boost_grid if seeds_boost_grid else [1.0]
    huber_opts = huber_delta_grid if (tune_robust and huber_delta_grid) else [0.0]
    lam_t_opts = lam_t_grid if (tune_robust and lam_t_grid) else [0.5]
    use_js_opts = [False, True] if tune_robust else [False]
    use_huber_opts = [False, True] if tune_robust else [False]
    bal_opts = [False, True] if tune_balance_by_juror else [balance_by_juror]
    cap_opts = cap_logmult_grid if cap_logmult_grid else [cap_logmult if cap_logmult is not None else 0.0]
    use_rel_opts = [False, True] if tune_reliability else [False]
    rel_alpha_opts = rel_alpha_grid if (tune_reliability and rel_alpha_grid) else [1.0]
    best = (float("inf"), lam_grid[0], p_grid[0], seeds_boost_opts[0], False, lam_t_opts[0], False, huber_opts[0])

    for lam in lam_grid:
        for p in p_grid:
            for sb in seeds_boost_opts:
                for use_js in use_js_opts:
                    for lam_t in lam_t_opts:
                        for use_huber in use_huber_opts:
                            for huber_delta in huber_opts:
                                if not use_huber and huber_delta != huber_opts[0]:
                                    continue
                                for balance_flag in bal_opts:
                                    for cap_val in cap_opts:
                                        cap_v = None if (isinstance(cap_val, float) and abs(cap_val) < 1e-12) else cap_val
                                        for use_rel in use_rel_opts:
                                            for rel_alpha in rel_alpha_opts:
                                                fold_costs = []
                                                for holdout in folds:
                                                    # Split samples
                                                    train_s = [s for s in samples if s.juror not in holdout]
                                                    val_s = [s for s in samples if s.juror in holdout]
                                                    if not train_s or not val_s:
                                                        continue
                                                    train_counts = counts_by_juror(train_s)
                                                    z0 = build_priors(repo_slugs, seed_set)
                                                    if use_js or use_huber:
                                                        z_cv, _ = solve_with_juror_scale_and_huber(
                                                            len(repo_slugs), train_s, train_counts, z0, lam,
                                                            repo_slugs, seed_set, p,
                                                            balance_by_juror=balance_flag,
                                                            seeds_boost=sb,
                                                            cap_logmult=cap_v,
                                                            use_huber=use_huber, huber_delta=huber_delta if use_huber else 1.0,
                                                            use_juror_scale=use_js, lam_t=lam_t,
                                                            t_clip=t_clip, verbose=False,
                                                            use_reliability=use_rel, rel_alpha=rel_alpha, rel_min=rel_min,
                                                            reliability_seeds_only=reliability_seeds_only,
                                                        )
                                                    else:
                                                        A, b, degree = build_normal_equations(
                                                            len(repo_slugs), train_s, train_counts, weight_power=p,
                                                            repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=sb,
                                                            balance_by_juror=balance_flag, cap_logmult=cap_v,
                                                        )
                                                        z_cv = solve_ridge(A, b, z0, degree, lam=lam)
                                                    val_eval = val_s
                                                    if seeds_only_eval:
                                                        val_eval = [s for s in val_s if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
                                                        if not val_eval:
                                                            continue
                                                    cost = evaluate_cost(z_cv, val_eval)
                                                    fold_costs.append(cost)
                                                if fold_costs:
                                                    avg = float(np.mean(fold_costs))
                                                    if avg < best[0]:
                                                        best = (avg, lam, p, sb, use_js, lam_t, use_huber, huber_delta)
    return best[1], best[2], best[3], best[4], best[5], best[6], best[7]


def fit_with_cv_features(
    samples: List[Sample],
    repo_slugs: List[str],
    seed_set: Set[str],
    X: np.ndarray,
    lam_z_grid: List[float],
    lam_theta_grid: List[float],
    p_grid: List[float],
    kfold: int = 5,
    rng_seed: int = 42,
    seeds_only_eval: bool = True,
    *,
    seeds_boost_grid: List[float] | None = None,
    tune_robust: bool = False,
    huber_delta_grid: List[float] | None = None,
    lam_t_grid: List[float] | None = None,
    t_clip: Tuple[float, float] = (0.8, 1.25),
    balance_by_juror: bool = False,
    cap_logmult: float | None = None,
    # New CV toggles
    tune_balance_by_juror: bool = False,
    cap_logmult_grid: List[float] | None = None,
    # Reliability reweighting
    tune_reliability: bool = False,
    rel_alpha_grid: List[float] | None = None,
    rel_min: float = 0.3,
    reliability_seeds_only: bool = True,
) -> Tuple[float, float, float, float, bool, float, bool, float, bool, float | None, bool, float]:
    """Juror-level CV for a feature-informed model.
    Returns (lam_z, lam_theta, p, seeds_boost, use_juror_scale, lam_t, use_huber, huber_delta).
    """
    jurors = sorted({s.juror for s in samples})
    folds = juror_kfold(jurors, kfold, rng_seed)
    seeds_boost_opts = seeds_boost_grid if seeds_boost_grid else [1.0]
    huber_opts = huber_delta_grid if (tune_robust and huber_delta_grid) else [0.0]
    lam_t_opts = lam_t_grid if (tune_robust and lam_t_grid) else [0.5]
    use_js_opts = [False, True] if tune_robust else [False]
    use_huber_opts = [False, True] if tune_robust else [False]
    bal_opts = [False, True] if tune_balance_by_juror else [balance_by_juror]
    cap_opts = cap_logmult_grid if cap_logmult_grid else [cap_logmult if cap_logmult is not None else 0.0]
    use_rel_opts = [False, True] if tune_reliability else [False]
    rel_alpha_opts = rel_alpha_grid if (tune_reliability and rel_alpha_grid) else [1.0]
    best = (float("inf"), lam_z_grid[0], lam_theta_grid[0], p_grid[0], seeds_boost_opts[0], False, lam_t_opts[0], False, huber_opts[0])
    for lam_z in lam_z_grid:
        for lam_theta in lam_theta_grid:
            for p in p_grid:
                for sb in seeds_boost_opts:
                    for use_js in use_js_opts:
                        for lam_t in lam_t_opts:
                            for use_huber in use_huber_opts:
                                for huber_delta in huber_opts:
                                    if not use_huber and huber_delta != huber_opts[0]:
                                        continue
                                    for balance_flag in bal_opts:
                                        for cap_val in cap_opts:
                                            cap_v = None if (isinstance(cap_val, float) and abs(cap_val) < 1e-12) else cap_val
                                            for use_rel in use_rel_opts:
                                                for rel_alpha in rel_alpha_opts:
                                                    fold_costs = []
                                                    for holdout in folds:
                                                        train_s = [s for s in samples if s.juror not in holdout]
                                                        val_s = [s for s in samples if s.juror in holdout]
                                                        if not train_s or not val_s:
                                                            continue
                                                        train_counts = counts_by_juror(train_s)
                                                        # weights for pairs
                                                        weights = []
                                                        for s in train_s:
                                                            w = max(abs(s.c) ** p, 1e-6) / max(1, train_counts.get(s.juror, 1))
                                                            weights.append(w)
                                                        theta = fit_theta_from_pairs(X, train_s, weights, lam_theta)
                                                        z0_feat = X @ theta
                                                        # Solve around feature prior
                                                        if use_js or use_huber:
                                                            z_cv, _ = solve_with_juror_scale_and_huber(
                                                                len(repo_slugs), train_s, train_counts, z0_feat, lam_z,
                                                                repo_slugs, seed_set, p,
                                                                balance_by_juror=balance_flag,
                                                                seeds_boost=sb,
                                                                cap_logmult=cap_v,
                                                                use_huber=use_huber, huber_delta=huber_delta if use_huber else 1.0,
                                                                use_juror_scale=use_js, lam_t=lam_t,
                                                                t_clip=t_clip, verbose=False,
                                                                use_reliability=use_rel, rel_alpha=rel_alpha, rel_min=rel_min,
                                                                reliability_seeds_only=reliability_seeds_only,
                                                            )
                                                        else:
                                                            A, b, degree = build_normal_equations(
                                                                len(repo_slugs), train_s, train_counts, weight_power=p,
                                                                repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=sb,
                                                                balance_by_juror=balance_flag, cap_logmult=cap_v,
                                                            )
                                                            z_cv = solve_ridge(A, b, z0_feat, degree, lam=lam_z)
                                                        val_eval = val_s
                                                        if seeds_only_eval:
                                                            val_eval = [s for s in val_s if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
                                                            if not val_eval:
                                                                continue
                                                        cost = evaluate_cost(z_cv, val_eval)
                                                        fold_costs.append(cost)
                                                    if fold_costs:
                                                        avg = float(np.mean(fold_costs))
                                                        if avg < best[0]:
                                                            best = (avg, lam_z, lam_theta, p, sb, use_js, lam_t, use_huber, huber_delta)
    return best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8]


# -----------------------------
# Submission writing
# -----------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    m = float(np.max(z))
    ez = np.exp(z - m)
    return ez / ez.sum()


def timestamped_filename(prefix: str = "submission") -> str:
    # Local time; format YYYYMMDD_HHMMSS
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.csv"


def write_submission(
    path: str,
    test_path: str,
    weights_by_slug: Dict[str, float],
):
    rows = []
    with open(test_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            repo = row.get("repo") or row.get("repo_a") or ""
            repo = str(repo).strip()
            if not repo:
                continue
            parent = row.get("parent", "ethereum")
            w = float(weights_by_slug.get(repo_slug(repo), 0.0))
            rows.append({"repo": repo, "parent": parent, "weight": f"{w:.16f}"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=["repo", "parent", "weight"])
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Deep Fund L1 model: ridge Bradley–Terry with priors")
    ap.add_argument("--train", default=os.path.join("dataset", "train.csv"))
    ap.add_argument("--test", default=os.path.join("dataset", "test.csv"))
    ap.add_argument("--seeds", default="seedRepos.json")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--exclude_nonseeds", action="store_true", help="Exclude non-seed repos (default: include)")
    ap.add_argument("--no_cv", action="store_true", help="Skip juror-level CV (use defaults)")
    ap.add_argument("--lambda_default", type=float, default=0.3)
    ap.add_argument("--weight_power_default", type=float, default=1.0)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--rng_seed", type=int, default=42, help="RNG seed for determinism (CV folds and any randomness)")
    # Feature-based prior options
    ap.add_argument("--metrics", default="external_metrics.json", help="Path to external metrics JSON (optional)")
    ap.add_argument("--fetch_metrics", action="store_true", help="Fetch GitHub metrics into --metrics path (needs internet)")
    ap.add_argument("--use_features", action="store_true", help="Use feature-informed prior and CV over (lam_z, lam_theta, p)")
    ap.add_argument("--lambda_theta_default", type=float, default=1.0)
    # Logging and ensembling
    ap.add_argument("--verbose", action="store_true", help="Print detailed progress and diagnostics")
    ap.add_argument("--no_ensemble", action="store_true", help="Disable logits ensemble; use single best model")
    # Category calibration
    ap.add_argument("--no_calibrate_cats", action="store_true", help="Disable category calibration LS step")
    ap.add_argument("--lam_cat", type=float, default=0.3, help="Ridge for category calibration")
    # Seeds-only weighting boost during training
    ap.add_argument("--seeds_boost", type=float, default=1.3, help="Multiply weights of seeds-only pairs during fitting")
    ap.add_argument("--tune_seeds_boost", action="store_true", help="Cross-validate seeds_boost on juror folds using seeds-only eval")
    ap.add_argument("--seeds_boost_grid", type=str, default="1.4,1.5,1.6,1.7,1.8", help="Comma-separated grid for seeds_boost when tuning")
    # Temperature calibration
    ap.add_argument("--no_temp_cal", action="store_true", help="Disable global temperature calibration on seeds-only pairs")
    # Juror balancing and capping
    ap.add_argument("--balance_by_juror", action="store_true", help="Balance sample weights by juror activity (each juror contributes ~equally)")
    ap.add_argument("--cap_logmult", type=float, default=4.5, help="Cap absolute log-multiplier |c| to this value (e.g., ln 90 ~= 4.5)")
    # Juror-scale and Huber IRLS
    ap.add_argument("--juror_scale", action="store_true", help="Learn per-juror scales t_j via alternating optimization (ridge toward 1)")
    ap.add_argument("--lam_t", type=float, default=0.5, help="Ridge regularization for juror scales t_j (toward 1)")
    ap.add_argument("--t_clip_lo", type=float, default=0.6, help="Lower clip for juror scale t_j")
    ap.add_argument("--t_clip_hi", type=float, default=1.6, help="Upper clip for juror scale t_j")
    ap.add_argument("--huber", action="store_true", help="Use Huber IRLS reweighting for robustness")
    ap.add_argument("--huber_delta", type=float, default=3.0, help="Huber delta parameter for robust reweighting (INCREASED: preserve extreme signals)")
    ap.add_argument("--tune_robust", action="store_true", help="Cross-validate robust options (juror_scale, lam_t) and Huber delta")
    ap.add_argument("--lam_t_grid", type=str, default="0.3,0.5,0.8", help="Comma-separated grid for lam_t when tuning robust options")
    ap.add_argument("--huber_delta_grid", type=str, default="2.0,3.0,4.0,5.0", help="Comma-separated grid for Huber delta (EXPANDED: NEW data has 19% extreme multipliers, preserve signal)")
    # Per-juror additive bias and reliability weighting
    ap.add_argument("--juror_bias", action="store_true", help="Learn per-juror additive bias b_j (ridge toward 0)")
    ap.add_argument("--lam_b", type=float, default=0.3, help="Ridge regularization for juror bias b_j (toward 0)")
    ap.add_argument("--reliability", action="store_true", help="Use per-juror reliability reweighting estimated from residuals")
    ap.add_argument("--rel_alpha", type=float, default=1.0, help="Reliability strength; higher downweights noisy jurors more")
    ap.add_argument("--rel_min", type=float, default=0.3, help="Minimum reliability weight per juror")
    ap.add_argument("--reliability_seeds_only", action="store_true", help="Estimate reliability using seeds-only pairs")
    # Optional CV toggles for balancing/cap (kept off by default)
    ap.add_argument("--tune_balance_by_juror", action="store_true", help="Cross-validate whether to balance by juror")
    ap.add_argument("--cap_logmult_grid", type=str, default="", help="Comma-separated grid for cap_logmult (use 0 for None)")
    # Reliability CV placeholders (not applied to final fit yet)
    ap.add_argument("--tune_reliability", action="store_true", help="Cross-validate reliability usage (experimental)")
    ap.add_argument("--rel_alpha_grid", type=str, default="", help="Comma-separated grid for rel_alpha when tuning reliability (experimental)")
    args = ap.parse_args()

    # Determinism
    try:
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)
    except Exception:
        pass

    # Load seeds and index repos
    seed_urls = read_seeds(args.seeds)
    seed_slugs = [repo_slug(u) for u in seed_urls]
    seed_set = set(seed_slugs)

    repo_slugs, repo_to_idx = index_repos(args.train, seed_urls, include_nonseeds=not args.exclude_nonseeds)

    # Read samples
    samples, _old_counts = read_train_samples(args.train, repo_to_idx)
    juror_counts = counts_by_juror(samples)

    # Optional metrics fetch/load
    if args.fetch_metrics:
        if args.verbose:
            print("[metrics] fetching GitHub metrics (respecting rate limits)...")
        maybe_fetch_github_metrics(repo_slugs, args.metrics)
    metrics = load_external_metrics(args.metrics)
    
    # IMPORTANT: Build priors FIRST to show OSO loading and apply boosts
    # Even if using features, this sets up the OSO data correctly
    if args.verbose:
        print(f"[priors] Building category priors with OSO boosts...")
    z0_base = build_priors(repo_slugs, seed_set)
    
    X, feat_names = build_feature_matrix(repo_slugs, metrics)
    if args.verbose:
        n_seeds = len(seed_set)
        n_total = len(repo_slugs)
        print(f"[data] repos total={n_total}, seeds={n_seeds}, jurors={len(juror_counts)}")
        # Basic c stats
        cs = np.array([abs(s.c) for s in samples], dtype=float)
        if cs.size:
            q = np.quantile(cs, [0.5, 0.9, 0.99])
            print(f"[data] |log-multiplier| median={q[0]:.3f} p90={q[1]:.3f} p99={q[2]:.3f}")

    # Hyperparameter selection
    # Collect candidate logits for potential ensembling
    candidate_names: List[str] = []
    candidate_logits: List[np.ndarray] = []

    if args.use_features:
        lam_z, lam_theta, p = args.lambda_default, args.lambda_theta_default, args.weight_power_default
        # Defaults for tuned extras
        seeds_boost_final = args.seeds_boost
        use_js_final = args.juror_scale
        lam_t_final = args.lam_t
        use_huber_final = args.huber
        huber_delta_final = args.huber_delta
        if not args.no_cv:
            lam_z_grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
            lam_theta_grid = [0.1, 0.3, 1.0, 3.0]
            p_grid = [0.0, 0.5, 0.75, 1.0, 1.25]
            sb_grid = [float(x) for x in args.seeds_boost_grid.split(',')] if args.tune_seeds_boost else None
            lam_t_grid = [float(x) for x in args.lam_t_grid.split(',')] if args.tune_robust else None
            hd_grid = [float(x) for x in args.huber_delta_grid.split(',')] if args.tune_robust else None
            # Optional grids (experimental)
            cap_grid = [float(x) for x in args.cap_logmult_grid.split(',')] if args.cap_logmult_grid else None
            # Reliability grid parsed but not applied to final fit yet
            rel_grid = [float(x) for x in args.rel_alpha_grid.split(',')] if (args.tune_reliability and args.rel_alpha_grid) else None
            lam_z, lam_theta, p, sb_cv, use_js_cv, lam_t_cv, use_huber_cv, hd_cv = fit_with_cv_features(
                samples, repo_slugs, seed_set, X,
                lam_z_grid, lam_theta_grid, p_grid,
                kfold=max(2, args.kfold), rng_seed=args.rng_seed, seeds_only_eval=True,
                seeds_boost_grid=sb_grid, tune_robust=args.tune_robust,
                huber_delta_grid=hd_grid, lam_t_grid=lam_t_grid,
                t_clip=(args.t_clip_lo, args.t_clip_hi),
                balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
                tune_balance_by_juror=args.tune_balance_by_juror,
                cap_logmult_grid=cap_grid,
                tune_reliability=False,
            )
            if args.tune_seeds_boost:
                seeds_boost_final = sb_cv
            if args.tune_robust:
                use_js_final = use_js_cv
                lam_t_final = lam_t_cv
                use_huber_final = use_huber_cv
                huber_delta_final = hd_cv
        # Fit on full data
        weights = []
        for s in samples:
            w = max(abs(s.c) ** p, 1e-6) / max(1, juror_counts.get(s.juror, 1))
            weights.append(w)
        theta = fit_theta_from_pairs(X, samples, weights, lam_theta)
        z0_feat_learned = X @ theta
        # Combine learned feature prior with OSO-boosted category prior
        # Use weighted average: 70% learned features + 30% OSO-boosted priors
        z0_feat = 0.7 * z0_feat_learned + 0.3 * z0_base
        if use_js_final or use_huber_final:
            z, scales = solve_with_juror_scale_and_huber(
                len(repo_slugs), samples, juror_counts, z0_feat, lam_z,
                repo_slugs, seed_set, p,
                balance_by_juror=args.balance_by_juror,
                seeds_boost=seeds_boost_final,
                cap_logmult=args.cap_logmult,
                use_huber=use_huber_final, huber_delta=huber_delta_final,
                use_juror_scale=use_js_final, lam_t=lam_t_final,
                t_clip=(args.t_clip_lo, args.t_clip_hi), verbose=args.verbose,
                use_reliability=args.reliability, rel_alpha=args.rel_alpha, rel_min=args.rel_min, reliability_seeds_only=args.reliability_seeds_only,
                use_bias=args.juror_bias, lam_b=args.lam_b,
            )
        else:
            A, b, degree = build_normal_equations(
                len(repo_slugs), samples, juror_counts, weight_power=p,
                repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=seeds_boost_final,
                balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
            )
            z = solve_ridge(A, b, z0_feat, degree, lam=lam_z)
        cv_msg = f"CV-picked lam_z={lam_z:.4f}, lam_theta={lam_theta:.4f}, p={p:.2f}"
        if not args.no_cv and (args.tune_seeds_boost or args.tune_robust):
            cv_msg += f" | seeds_boost={seeds_boost_final:.2f}"
            if args.tune_robust:
                cv_msg += f" | juror_scale={use_js_final} lam_t={lam_t_final:.3f} huber={use_huber_final} delta={huber_delta_final:.2f}"
        print(cv_msg)
        candidate_names.append("feature_ridge" + ("_js" if use_js_final else "") + ("_huber" if use_huber_final else ""))
        candidate_logits.append(z.copy())
    else:
        lam, p = args.lambda_default, args.weight_power_default
        # Defaults for tuned extras
        seeds_boost_final = args.seeds_boost
        use_js_final = args.juror_scale
        lam_t_final = args.lam_t
        use_huber_final = args.huber
        huber_delta_final = args.huber_delta
        if not args.no_cv:
            lam_grid = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.0, 3.0]
            p_grid = [0.0, 0.5, 0.75, 1.0, 1.25]
            sb_grid = [float(x) for x in args.seeds_boost_grid.split(',')] if args.tune_seeds_boost else None
            lam_t_grid = [float(x) for x in args.lam_t_grid.split(',')] if args.tune_robust else None
            hd_grid = [float(x) for x in args.huber_delta_grid.split(',')] if args.tune_robust else None
            # Optional grids (experimental)
            cap_grid = [float(x) for x in args.cap_logmult_grid.split(',')] if args.cap_logmult_grid else None
            rel_grid = [float(x) for x in args.rel_alpha_grid.split(',')] if (args.tune_reliability and args.rel_alpha_grid) else None
            lam, p, sb_cv, use_js_cv, lam_t_cv, use_huber_cv, hd_cv = fit_with_cv(
                samples, repo_slugs, seed_set, lam_grid, p_grid,
                kfold=max(2, args.kfold), rng_seed=args.rng_seed, seeds_only_eval=True,
                seeds_boost_grid=sb_grid, tune_robust=args.tune_robust,
                huber_delta_grid=hd_grid, lam_t_grid=lam_t_grid,
                t_clip=(args.t_clip_lo, args.t_clip_hi),
                balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
                tune_balance_by_juror=args.tune_balance_by_juror,
                cap_logmult_grid=cap_grid,
                tune_reliability=False,
            )
            if args.tune_seeds_boost:
                seeds_boost_final = sb_cv
            if args.tune_robust:
                use_js_final = use_js_cv
                lam_t_final = lam_t_cv
                use_huber_final = use_huber_cv
                huber_delta_final = hd_cv
        # Fit on full data
        z0 = build_priors(repo_slugs, seed_set)
        if use_js_final or use_huber_final:
            z, scales = solve_with_juror_scale_and_huber(
                len(repo_slugs), samples, juror_counts, z0, lam,
                repo_slugs, seed_set, p,
                balance_by_juror=args.balance_by_juror,
                seeds_boost=seeds_boost_final,
                cap_logmult=args.cap_logmult,
                use_huber=use_huber_final, huber_delta=huber_delta_final,
                use_juror_scale=use_js_final, lam_t=lam_t_final,
                t_clip=(args.t_clip_lo, args.t_clip_hi), verbose=args.verbose,
                use_reliability=args.reliability, rel_alpha=args.rel_alpha, rel_min=args.rel_min, reliability_seeds_only=args.reliability_seeds_only,
                use_bias=args.juror_bias, lam_b=args.lam_b,
            )
        else:
            A, b, degree = build_normal_equations(
                len(repo_slugs), samples, juror_counts, weight_power=p,
                repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=seeds_boost_final,
                balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
            )
            z = solve_ridge(A, b, z0, degree, lam=lam)
        if not args.no_cv and (args.tune_seeds_boost or args.tune_robust):
            print(f"CV-picked lam={lam:.4f}, p={p:.2f} | seeds_boost={seeds_boost_final:.2f}" + (f" | juror_scale={use_js_final} lam_t={lam_t_final:.3f} huber={use_huber_final} delta={huber_delta_final:.2f}" if args.tune_robust else ""))
        candidate_names.append("category_ridge" + ("_js" if use_js_final else "") + ("_huber" if use_huber_final else ""))
        candidate_logits.append(z.copy())

    # Also produce a seeds-only candidate to align closer with leaderboard
    # Build equations using only seed-seed pairs, solve around the same prior used above
    train_s_seeds = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
    if train_s_seeds:
        train_counts_seed = counts_by_juror(train_s_seeds)
        if args.use_features:
            if ('use_js_final' in locals() and use_js_final) or ('use_huber_final' in locals() and use_huber_final):
                z_seed, _ = solve_with_juror_scale_and_huber(
                    len(repo_slugs), train_s_seeds, train_counts_seed, z0_feat, lam_z,
                    repo_slugs, seed_set, p,
                    balance_by_juror=args.balance_by_juror,
                    seeds_boost=(seeds_boost_final if 'seeds_boost_final' in locals() else args.seeds_boost),
                    cap_logmult=args.cap_logmult,
                    use_huber=(use_huber_final if 'use_huber_final' in locals() else args.huber), huber_delta=(huber_delta_final if 'huber_delta_final' in locals() else args.huber_delta),
                    use_juror_scale=(use_js_final if 'use_js_final' in locals() else args.juror_scale), lam_t=(lam_t_final if 'lam_t_final' in locals() else args.lam_t),
                    t_clip=(args.t_clip_lo, args.t_clip_hi), verbose=args.verbose,
                    use_reliability=args.reliability, rel_alpha=args.rel_alpha, rel_min=args.rel_min, reliability_seeds_only=args.reliability_seeds_only,
                    use_bias=args.juror_bias, lam_b=args.lam_b,
                )
            else:
                A_seed, b_seed, degree_seed = build_normal_equations(
                    len(repo_slugs), train_s_seeds, train_counts_seed, weight_power=p,
                    repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=(seeds_boost_final if 'seeds_boost_final' in locals() else args.seeds_boost),
                    balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
                )
                z_seed = solve_ridge(A_seed, b_seed, z0_feat, degree_seed, lam=lam_z)
        else:
            if ('use_js_final' in locals() and use_js_final) or ('use_huber_final' in locals() and use_huber_final):
                z0 = build_priors(repo_slugs, seed_set)
                z_seed, _ = solve_with_juror_scale_and_huber(
                    len(repo_slugs), train_s_seeds, train_counts_seed, z0, lam,
                    repo_slugs, seed_set, p,
                    balance_by_juror=args.balance_by_juror,
                    seeds_boost=(seeds_boost_final if 'seeds_boost_final' in locals() else args.seeds_boost),
                    cap_logmult=args.cap_logmult,
                    use_huber=(use_huber_final if 'use_huber_final' in locals() else args.huber), huber_delta=(huber_delta_final if 'huber_delta_final' in locals() else args.huber_delta),
                    use_juror_scale=(use_js_final if 'use_js_final' in locals() else args.juror_scale), lam_t=(lam_t_final if 'lam_t_final' in locals() else args.lam_t),
                    t_clip=(args.t_clip_lo, args.t_clip_hi), verbose=args.verbose,
                    use_reliability=args.reliability, rel_alpha=args.rel_alpha, rel_min=args.rel_min, reliability_seeds_only=args.reliability_seeds_only,
                    use_bias=args.juror_bias, lam_b=args.lam_b,
                )
            else:
                A_seed, b_seed, degree_seed = build_normal_equations(
                    len(repo_slugs), train_s_seeds, train_counts_seed, weight_power=p,
                    repo_slugs=repo_slugs, seed_set=seed_set, seeds_boost=(seeds_boost_final if 'seeds_boost_final' in locals() else args.seeds_boost),
                    balance_by_juror=args.balance_by_juror, cap_logmult=args.cap_logmult,
                )
                z0 = build_priors(repo_slugs, seed_set)
                z_seed = solve_ridge(A_seed, b_seed, z0, degree_seed, lam=lam)
        candidate_names.append("seeds_only_ridge" + ("_js" if (('use_js_final' in locals() and use_js_final) or (not 'use_js_final' in locals() and args.juror_scale)) else "") + ("_huber" if (('use_huber_final' in locals() and use_huber_final) or (not 'use_huber_final' in locals() and args.huber)) else ""))
        candidate_logits.append(z_seed.copy())

    # Add the pure prior as a candidate
    if args.use_features:
        candidate_names.append("feature_prior_only")
        candidate_logits.append(z0_feat.copy())
    else:
        z0 = build_priors(repo_slugs, seed_set)
        candidate_names.append("category_prior_only")
        candidate_logits.append(z0.copy())

    # Compute diagnostics per candidate
    if args.verbose:
        print("[candidates] diagnostics:")
        for nm, zz in zip(candidate_names, candidate_logits):
            sse_all = evaluate_cost(zz, samples)
            mse_all = sse_all / max(1, len(samples))
            seeds_pairs = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
            sse_seed = evaluate_cost(zz, seeds_pairs) if seeds_pairs else float('nan')
            mse_seed = sse_seed / max(1, len(seeds_pairs)) if seeds_pairs else float('nan')
            print(f"  - {nm:22s} SSE={sse_all:.3f} MSE={mse_all:.6f} | Seeds-SSE={sse_seed:.3f} Seeds-MSE={mse_seed:.6f}")

    # Optionally ensemble candidates by optimizing simplex weights to minimize SSE
    def build_D_and_y(L_list: List[np.ndarray], samples_list: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        K = len(L_list)
        D = np.zeros((len(samples_list), K), dtype=float)
        y = np.zeros(len(samples_list), dtype=float)
        for i, s in enumerate(samples_list):
            for k in range(K):
                D[i, k] = L_list[k][s.b] - L_list[k][s.a]
            y[i] = s.c
        return D, y

    def project_simplex(v: np.ndarray) -> np.ndarray:
        # Duchi et al. projection onto simplex {w>=0, sum w = 1}
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
        if rho.size == 0:
            tau = 0.0
        else:
            rho = rho[-1]
            tau = (cssv[rho] - 1.0) / (rho + 1)
        w = np.maximum(v - tau, 0.0)
        # Numerical normalization
        s = w.sum()
        if s <= 0:
            return np.full_like(v, 1.0 / n)
        return w / s

    def optimize_simplex_weights(L_list: List[np.ndarray], samples_list: List[Sample]) -> np.ndarray:
        D, y = build_D_and_y(L_list, samples_list)
        K = D.shape[1]
        # Try closed-form LS then project
        try:
            w_ls = np.linalg.lstsq(D, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            w_ls = np.full(K, 1.0 / K)
        w = project_simplex(w_ls)
        # Refine by projected gradient descent
        # Lipschitz constant for gradient of ||Dw - y||^2 is 2 * ||D||_2^2
        # Approx with column norm upper bound
        Lg = 2.0 * float(np.max(np.sum(D * D, axis=0)))
        step = 1.0 / max(Lg, 1e-6)
        for _ in range(200):
            grad = 2.0 * (D.T @ (D @ w - y))
            w = project_simplex(w - step * grad)
        return w

    use_ensemble = not args.no_ensemble and len(candidate_logits) >= 2
    if use_ensemble:
        if args.verbose:
            print(f"[ensemble] optimizing weights over {len(candidate_logits)} candidates: {candidate_names}")
        seeds_pairs = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
        w = optimize_simplex_weights(candidate_logits, seeds_pairs if seeds_pairs else samples)
        if args.verbose:
            combo = ", ".join(f"{nm}:{wk:.3f}" for nm, wk in zip(candidate_names, w))
            print(f"[ensemble] weights: {combo}")
        z_combo = np.zeros_like(candidate_logits[0])
        for wk, zz in zip(w, candidate_logits):
            z_combo += wk * zz
        z = z_combo
    # Category calibration step (seeds-only by default)
    if not args.no_calibrate_cats:
        z_before = z.copy()
        z, alpha, cat_names = calibrate_category_offsets(repo_slugs, samples, seed_set, z, lam_cat=args.lam_cat, seeds_only=True)
        if args.verbose:
            al = ", ".join(f"{c}:{a:+.4f}" for c, a in zip(cat_names, alpha))
            # show improvement in seeds-only SSE
            seeds_pairs = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
            if seeds_pairs:
                sse_before = evaluate_cost(z_before, seeds_pairs)
                sse_after = evaluate_cost(z, seeds_pairs)
                print(f"[calib] category offsets: {al}")
                print(f"[calib] seeds-only SSE before={sse_before:.4f} after={sse_after:.4f}")
    # Temperature calibration
    if not args.no_temp_cal:
        z_before = z.copy()
        z, t = calibrate_temperature(repo_slugs, samples, seed_set, z, seeds_only=True)
        if args.verbose:
            seeds_pairs = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
            if seeds_pairs:
                sse_before = evaluate_cost(z_before, seeds_pairs)
                sse_after = evaluate_cost(z, seeds_pairs)
                print(f"[calib] temperature t={t:.4f}")
                print(f"[calib] seeds-only SSE before={sse_before:.4f} after={sse_after:.4f}")
    # Report train SSE/MSE (all pairs and seeds-only)
    train_sse = evaluate_cost(z, samples)
    train_mse = train_sse / max(1, len(samples))
    seeds_pairs = [s for s in samples if (repo_slugs[s.a] in seed_set and repo_slugs[s.b] in seed_set)]
    if seeds_pairs:
        seeds_sse = evaluate_cost(z, seeds_pairs)
        seeds_mse = seeds_sse / max(1, len(seeds_pairs))
        print(f"Train SSE={train_sse:.4f}  MSE={train_mse:.6f}  |  Seeds-only SSE={seeds_sse:.4f}  MSE={seeds_mse:.6f}")
    else:
        print(f"Train SSE={train_sse:.4f}  MSE={train_mse:.6f}")

    # Convert to probabilities and restrict to seeds from test.csv
    probs_all = softmax(z)

    # Build mapping slug->prob for seeds
    weights_by_slug: Dict[str, float] = {}
    # We'll re-normalize to sum to 1 over the 45 seed repos listed in test.csv
    test_rows: List[str] = []
    with open(args.test, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            repo = str(row.get("repo", "")).strip()
            if repo:
                test_rows.append(repo)
    seed_probs = []
    for repo in test_rows:
        s = repo_slug(repo)
        idx = repo_to_idx.get(s)
        pval = float(probs_all[idx]) if idx is not None else 0.0
        seed_probs.append((s, pval))
    denom = sum(p for _, p in seed_probs) or 1.0
    for s, pval in seed_probs:
        weights_by_slug[s] = pval / denom

    if args.verbose:
        # Top 10 seeds by weight
        top = sorted(((repo, w) for repo, w in weights_by_slug.items()), key=lambda x: -x[1])[:10]
        print("[final] top-10 seeds:")
        for repo, w in top:
            print(f"  {repo:45s}  {w:.4f}")
        # Category totals
        cat_sum: Dict[str, float] = Counter()
        for repo, w in weights_by_slug.items():
            cat_sum[categorize(repo)] += w
        print("[final] category totals:")
        for cat in ["EL", "CL", "TOOLS", "LANG", "SPECS", "INFRA"]:
            print(f"  {cat:6s} {cat_sum.get(cat, 0.0):.4f}")

    # Write submission
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, timestamped_filename("submission"))
    write_submission(out_path, args.test, weights_by_slug)
    print(f"Wrote {out_path}")
    
    # Save base weights for Phase 1 juror corrections
    weights_path = os.path.join(args.outdir, "base_weights.json")
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(weights_by_slug, f, indent=2)
    print(f"Saved base weights to {weights_path}")


if __name__ == "__main__":
    main()
