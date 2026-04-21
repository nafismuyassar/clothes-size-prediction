import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Baca data training
train = pd.read_csv('data/prepared/train.csv')

# Ganti 'size' dengan nama kolom target di dataset Anda
target = 'size'
X = train.drop(columns=[target])
y = train[target]

# Ubah data kategori menjadi angka
X = pd.get_dummies(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Simpan model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("Model berhasil disimpan di models/model.pkl")