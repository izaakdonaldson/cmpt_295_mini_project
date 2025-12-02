import glob
import pandas as pd
import matplotlib.pyplot as plt

# Get all CSV files in the folder
#csv_files = glob.glob("*.csv")

#files = ["co_block_matmul.csv","block_matmul_L1.csv", "block_matmul_L2.csv", "block_matmul_L3.csv", "block_matmul_no_cache.csv"]
#files = ["intel/co_block_matmul.csv", "intel/block_matmul_L1.csv", "intel/block_matmul_L2.csv", "intel/block_matmul_L3.csv", "intel/block_matmul_no_cache.csv"]
#files = ["simd_matmul.csv", "mt_matmul.csv", "mt_simd_matmul.csv", "block_matmul_L1.csv", "co_block_matmul.csv", "cuda_matmul.csv"]
#files = ["naive_matmul.csv", "simd_matmul.csv", "mt_matmul.csv", "mt_simd_matmul.csv"]
files = ["mt_simd_matmul.csv", "co_block_matmul.csv", "cuda_matmul.csv"]

plt.figure()

all_x = []

for f in files:
    df_input = pd.read_csv(f)
    df = df_input.tail(4)
    all_x.extend(df['size'])
    plt.plot(df['size'], df['time_ms'], marker='o', markersize=4, label=f)

plt.xscale("log")


ax = plt.gca()
ax.set_xticks([])
ax.set_xticks([], minor=True)


ax.set_xticks(all_x)
ax.set_xticklabels(all_x, rotation=45)

plt.xlabel("Matrix Size NxN (log scale)")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()
