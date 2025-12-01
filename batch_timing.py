import subprocess
import csv

sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

csv_file = "csv_output/mt_simd_matmul.csv"

# write CSV header
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["size", "time_ms"])  # header

# run each size and append result
for n in sizes:
    result = subprocess.check_output(["./matmul_bench", str(n)])
    line = result.decode().strip()  # e.g. "256,13.52"
    size_str, time_str = line.split(",")

    # Print real-time update
    print(f"N={size_str}, time={time_str} ms")

    # Append to CSV
    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([int(size_str), float(time_str)])

print("Saved results to", csv_file)
