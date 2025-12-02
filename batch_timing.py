import subprocess
import csv

sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] #8192, 16384

csv_file = "csv_output/extra/cuda_matmul.csv"

with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["size", "time_ms"])

for n in sizes:
    result = subprocess.check_output(["./matmul_bench", str(n)])
    line = result.decode().strip()
    size_str, time_str = line.split(",")

    print(f"N={size_str}, time={time_str} ms")

    with open(csv_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([int(size_str), float(time_str)])

print("Saved results to", csv_file)
