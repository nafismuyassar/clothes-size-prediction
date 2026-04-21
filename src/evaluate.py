import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

# Baca data test
test = pd.read_csv('data/prepared/test.csv')
target = 'size'
X_test = test.drop(columns=[target])
y_test = test[target]

# Ubah data kategori menjadi angka
X_test = pd.get_dummies(X_test)

# Load model
model = joblib.load('models/model.pkl')
le = joblib.load('models/label_encoder.pkl')
y_test_encoded = le.transform(y_test)

# Prediksi
y_pred = model.predict(X_test)

# Hitung akurasi
acc = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

# Simpan hasil ke file JSON
metrics = {'accuracy': acc, 'f1_score': f1}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Akurasi: {acc:.4f}, F1-Score: {f1:.4f}")
print("Hasil disimpan di metrics.json")