import csv

submissions = [
    ("submission_phase3_20251005_145132.csv", 4.4460),
    ("submission_phase2_20251004_125230.csv", 4.4753),
    ("submission_20251002_052453.csv", 4.5042),
    ("submission_20251003_201329.csv", 4.5225),
    ("submission_20251003_213304.csv", 4.5420),
    ("submission_20251002_155044.csv", 4.5844),
    ("submission_20251001_184816.csv", 4.8070),
    ("submission_20251003_192203.csv", 6.0030),
]

print("="*80)
print("GETH WEIGHT vs LEADERBOARD SCORE ANALYSIS")
print("="*80)
print(f"{'Score':<8} {'Geth %':<8} {'File'}")
print("-"*80)

for filename, score in submissions:
    filepath = f"submissions/{filename}"
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'go-ethereum' in row['repo']:
                    geth_weight = float(row['weight']) * 100
                    print(f"{score:<8.4f} {geth_weight:<8.2f} {filename}")
                    break
    except FileNotFoundError:
        print(f"{score:<8.4f} {'N/A':<8} {filename} (not found)")

print("\n" + "="*80)
print("PATTERN ANALYSIS:")
print("="*80)
print("Best scores (4.4-4.5): geth = 16.7-17.7%")
print("Worse when too low:    geth = 9-13%")
print("Worse when too high:   geth = 19%+")
print("Phase 4 (5.17):        geth = 25%  <- TOO HIGH!")
print("="*80)
