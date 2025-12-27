import os
import re
import csv
from pathlib import Path
from collections import defaultdict

# Base path
base_path = Path(r"C:\AR team\Ali\Bitbucket\olsolve")

# Find all THREAD folders
thread_folders = sorted([f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("THREAD")])

# Pattern to match the result lines
# Example: > C:\AR team\Ali\MPS_files\mps_to_test_new_cuts_heurs\no_700.MPS optimal 6.70317e+07 6.70317e+07 0, time = 11.38 secs
prefix = r"> C:\AR team\Ali\MPS_files\mps_to_test_new_cuts_heurs\\"
pattern = re.compile(r'^> C:\\AR team\\Ali\\MPS_files\\mps_to_test_new_cuts_heurs\\(.+?)\.MPS\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+),\s*time\s*=\s*(\S+)\s*secs', re.IGNORECASE)

# Store results: {model_name: {thread_name: (status, solution, bound, gap, time)}}
results = defaultdict(dict)
thread_names = []

for folder in thread_folders:
    log_file = folder / "optonomy.log"
    thread_name = folder.name
    thread_names.append(thread_name)

    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith("> C:\\AR team\\Ali\\MPS_files\\mps_to_test_new_cuts_heurs\\"):
                    match = pattern.match(line.strip())
                    if match:
                        model_name = match.group(1) + ".MPS"
                        status = match.group(2)
                        solution = match.group(3)
                        bound = match.group(4)
                        gap = match.group(5)
                        time_secs = match.group(6)
                        results[model_name][thread_name] = {
                            'status': status,
                            'solution': solution,
                            'bound': bound,
                            'gap': gap,
                            'time': time_secs
                        }

# Short names
short_names = {
    "THREAD 0": "T0",
    "THREAD 1 - Lift Project": "T1-Lift",
    "THREAD 2 - Feasibility pump": "T2-FP",
    "THREAD 3 - RINS AND RENS": "T3-RR",
    "THREAD 3a - RINS": "T3a-RINS",
    "THREAD 3b - RENS": "T3b-RENS",
    "THREAD 4 wrong": "T4-wrong"
}

# Get all model names
all_models = sorted(results.keys())

# Save to CSV file
output_dir = Path(r"C:\AR team\Ali\ocr_with_kraken_public-main")

# Combined CSV with all metrics
csv_file = output_dir / "comparison_all.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Header row 1: Thread names spanning 4 columns each
    header1 = [""]
    for tn in thread_names:
        short = short_names.get(tn, tn)
        header1.extend([short, "", "", ""])
    writer.writerow(header1)

    # Header row 2: Metric names
    header2 = ["Model"]
    for tn in thread_names:
        header2.extend(["Solution", "Bound", "Gap", "Time"])
    writer.writerow(header2)

    # Data rows
    for model in all_models:
        row = [model]
        for tn in thread_names:
            if tn in results[model]:
                r = results[model][tn]
                row.extend([r['solution'], r['bound'], r['gap'], r['time']])
            else:
                row.extend(["", "", "", ""])
        writer.writerow(row)

print(f"Saved combined comparison to: {csv_file}")

# Print summary to console
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total models tested: {len(all_models)}")
print(f"Thread folders: {len(thread_names)}")
for tn in thread_names:
    count = sum(1 for m in all_models if tn in results[m])
    print(f"  {short_names.get(tn, tn)}: {count} models")

# Print a sample of results for quick view
print("\n" + "=" * 80)
print("SAMPLE: First 20 models - TIME (seconds)")
print("=" * 80)

# Header
header = f"{'Model':<60}"
for tn in thread_names:
    short = short_names.get(tn, tn)
    header += f" | {short:>10}"
print(header)
print("-" * len(header))

for model in all_models[:20]:
    row = f"{model:<60}"
    for tn in thread_names:
        if tn in results[model]:
            time_val = results[model][tn]['time']
            # Shorten the time display
            try:
                t = float(time_val)
                time_str = f"{t:.2f}"
            except:
                time_str = time_val[:8]
            row += f" | {time_str:>10}"
        else:
            row += f" | {'N/A':>10}"
    print(row)
