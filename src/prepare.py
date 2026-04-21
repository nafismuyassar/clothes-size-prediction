# src/prepare.py
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Baca data mentah
df = pd.read_csv('data/raw/clothes-size-prediction.csv')

# Contoh preprocessing sederhana:
# 1. Hapus baris dengan missing value (jika ada)
df.dropna(inplace=True)

# 2. Split data menjadi train dan test (80:20)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 3. Simpan hasil split
os.makedirs('data/prepared', exist_ok=True)
train.to_csv('data/prepared/train.csv', index=False)
test.to_csv('data/prepared/test.csv', index=False)

# 4. Buat plot distribusi sederhana (opsional, contoh)
plt.figure()
df.hist(figsize=(10,8))
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/data_distribution.png')
print("Plot distribusi data disimpan di plots/data_distribution.png")
print("Data berhasil di-split dan disimpan di folder data/prepared/")