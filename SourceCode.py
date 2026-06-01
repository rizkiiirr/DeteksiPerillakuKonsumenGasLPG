"""
========================================================
  DETEKSI ANOMALI PERILAKU PEMBELIAN LPG 3 KG
  ANN + SMOTE (Synthetic Minority Over-sampling Technique)
  Implementasi SMOTE dari scratch — tanpa library eksternal
========================================================
Dibuat untuk keperluan tugas akhir / penelitian.
Model  : Multilayer Perceptron (MLP) — bukan turunan ANN
Dataset: dataset_lpg_pivot.csv (214 sampel, 23 fitur)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score,
                              precision_score, recall_score, f1_score)


# ─────────────────────────────────────────────────────────────
# FUNGSI SMOTE (Chawla et al., 2002) — Implementasi Manual
# ─────────────────────────────────────────────────────────────
def smote(X, y, minority_class=1, k=5, random_state=42):
    """
    SMOTE: Synthetic Minority Over-sampling Technique

    Algoritma:
    1. Identifikasi semua sampel kelas minoritas
    2. Untuk tiap sampel minoritas, cari k tetangga terdekat (KNN)
    3. Pilih satu tetangga secara acak
    4. Buat sampel sintetis:  x_new = x_i + α × (x_nn - x_i)
       di mana α ∈ [0,1] dipilih secara acak (interpolasi)
    5. Ulangi hingga kelas seimbang

    Parameters:
        X              : array fitur (sudah ternormalisasi)
        y              : array label
        minority_class : label kelas minoritas (default 1)
        k              : jumlah tetangga terdekat (default 5)
        random_state   : seed untuk reprodusibilitas

    Returns:
        X_resampled, y_resampled, n_synthetic
    """
    rng = np.random.RandomState(random_state)

    X_min  = X[y == minority_class]   # sampel kelas minoritas
    n_min  = len(X_min)
    n_maj  = (y != minority_class).sum()
    n_need = n_maj - n_min             # berapa sampel sintetis dibutuhkan

    if n_need <= 0:
        return X, y, 0                 # sudah seimbang, tidak perlu SMOTE

    # Step 1 — Cari k nearest neighbors untuk setiap sampel minoritas
    knn = NearestNeighbors(n_neighbors=k + 1)  # +1: sampel itu sendiri
    knn.fit(X_min)
    neighbors = knn.kneighbors(X_min, return_distance=False)[:, 1:]

    # Step 2 — Buat sampel sintetis
    synthetic = []
    for _ in range(n_need):
        idx     = rng.randint(0, n_min)         # pilih satu sampel acak
        nn_idx  = neighbors[idx][rng.randint(0, k)]  # pilih satu tetangga
        alpha   = rng.random()                  # interpolation factor ∈ [0,1]
        x_new   = X_min[idx] + alpha * (X_min[nn_idx] - X_min[idx])
        synthetic.append(x_new)

    X_syn = np.array(synthetic)
    y_syn = np.full(n_need, minority_class)

    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, y_syn])

    return X_res, y_res, n_need


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('dataset_lpg.csv', sep=';')

print("=" * 65)
print("   DETEKSI ANOMALI LPG 3 KG — ANN + SMOTE")
print("=" * 65)
print(f"\n[DATASET]")
print(f"  Total sampel  : {len(df)}")
print(f"  Jumlah fitur  : {df.shape[1]-1}")
print(f"  Label Normal  (0): {(df['label']==0).sum()} sampel "
      f"({(df['label']==0).mean()*100:.1f}%)")
print(f"  Label Anomali (1): {(df['label']==1).sum()} sampel "
      f"({(df['label']==1).mean()*100:.1f}%)")

X = df.drop('label', axis=1).values
y = df['label'].values

# ─────────────────────────────────────────────────────────────
# 2. NORMALISASI — Min-Max Scaling ke rentang [0, 1]
# ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────
# 3. SPLIT DATA — 80% Train / 20% Test (Stratified)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n[DATA SPLIT]")
print(f"  Training : {len(X_train)} sampel")
print(f"  Testing  : {len(X_test)} sampel")

# ─────────────────────────────────────────────────────────────
# 4. TERAPKAN SMOTE PADA DATA TRAINING
#    PENTING: SMOTE hanya diterapkan pada data training, BUKAN testing
#    Menerapkan SMOTE pada test set akan mencemari evaluasi (data leakage)
# ─────────────────────────────────────────────────────────────
print(f"\n[SMOTE]")
print(f"  Sebelum — Normal: {(y_train==0).sum()}, "
      f"Anomali: {(y_train==1).sum()}")

X_train_smote, y_train_smote, n_syn = smote(
    X_train, y_train, minority_class=1, k=5, random_state=42)

print(f"  Sesudah — Normal: {(y_train_smote==0).sum()}, "
      f"Anomali: {(y_train_smote==1).sum()} "
      f"(+{n_syn} sampel sintetis)")
print(f"  Total training setelah SMOTE: {len(X_train_smote)} sampel")

# ─────────────────────────────────────────────────────────────
# 5. BANGUN & LATIH MODEL ANN + SMOTE
#    Arsitektur: Input(23) → Dense(64,ReLU) → Dense(32,ReLU) → Output(Sigmoid)
# ─────────────────────────────────────────────────────────────
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False
)

print(f"\n[TRAINING] Melatih model ANN dengan data SMOTE...")
model.fit(X_train_smote, y_train_smote)
print(f"[TRAINING] Selesai. Total iterasi: {model.n_iter_}")

# ─────────────────────────────────────────────────────────────
# 6. EVALUASI MODEL
# ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 65)
print("   HASIL EVALUASI — ANN + SMOTE")
print("=" * 65)
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  Precision   : {prec*100:.2f}%")
print(f"  Recall      : {rec*100:.2f}%")
print(f"  F1-Score    : {f1*100:.2f}%")
print(f"  AUC-ROC     : {auc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
print()
print(classification_report(y_test, y_pred,
      target_names=['Normal (0)', 'Anomali (1)']))

# ─────────────────────────────────────────────────────────────
# 7. 10-FOLD CROSS VALIDATION
#    SMOTE diterapkan di dalam setiap fold (best practice)
#    agar tidak terjadi data leakage dari fold validasi
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("   10-FOLD CROSS VALIDATION (SMOTE per fold)")
print("=" * 65)

cv_scores = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    Xtr, Xv = X_scaled[train_idx], X_scaled[val_idx]
    ytr, yv = y[train_idx], y[val_idx]

    # SMOTE hanya pada data training tiap fold
    Xtr_s, ytr_s, _ = smote(Xtr, ytr, k=5, random_state=42)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
        alpha=0.001, learning_rate='adaptive', max_iter=500, random_state=42)
    clf.fit(Xtr_s, ytr_s)

    score = accuracy_score(yv, clf.predict(Xv))
    cv_scores.append(score)
    print(f"  Fold {fold+1:2d}: {score*100:.2f}%")

cv_arr = np.array(cv_scores)
print(f"\n  Mean Accuracy : {cv_arr.mean()*100:.2f}%")
print(f"  Std Deviation : {cv_arr.std()*100:.2f}%")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 8. PREDIKSI INDIVIDUAL — contoh penggunaan
# ─────────────────────────────────────────────────────────────
print("\n[CONTOH PREDIKSI 5 SAMPEL TEST]")
print(f"{'No':>3} | {'Aktual':>7} | {'Prediksi':>9} | {'Prob Anomali':>12} | Status")
print("-" * 55)
for i in range(5):
    true = y_test[i]; pred = y_pred[i]; prob = y_prob[i]
    label  = "Anomali" if pred == 1 else "Normal "
    status = "✓ Benar" if true == pred else "✗ Salah"
    print(f"{i+1:>3} | {true:>7} | {label:>9} | {prob:>12.4f} | {status}")

print("\n[INFO] Untuk prediksi data baru:")
print("[INFO]   X_new_scaled = scaler.transform(X_new)")
print("[INFO]   pred = model.predict(X_new_scaled)")
print("[INFO]   prob = model.predict_proba(X_new_scaled)[:, 1]")
