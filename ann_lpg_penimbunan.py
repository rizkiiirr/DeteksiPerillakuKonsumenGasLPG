# =============================================================================
#  DETEKSI ANOMALI PERILAKU PEMBELIAN LPG 3 KG
#  Mengidentifikasi Indikasi Penimbunan oleh Pembeli Non-UMKM
#  Menggunakan Artificial Neural Network (Pure NumPy)
#
#  Model: ANN Vanilla (bukan turunan ANN)
#  Arsitektur: Input(10) → Hidden1(16, ReLU) → Hidden2(8, ReLU) → Output(1, Sigmoid)
#  Optimizer: Mini-batch Stochastic Gradient Descent
#  Loss: Binary Cross-Entropy (dengan class weighting)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. KELAS ANN (Pure NumPy — bukan turunan ANN seperti CNN/RNN/LSTM)
# ===========================================================================

class ArtificialNeuralNetwork:
    """
    Implementasi Artificial Neural Network murni menggunakan NumPy.

    Arsitektur: Fully Connected (Dense) Feedforward Neural Network
    - Aktivasi hidden layer : ReLU
    - Aktivasi output layer : Sigmoid (untuk klasifikasi biner)
    - Loss function         : Binary Cross-Entropy + L2 Regularization
    - Optimizer             : Mini-batch Stochastic Gradient Descent (SGD)
    - Inisialisasi bobot    : He Initialization (optimal untuk ReLU)

    Parameters
    ----------
    layer_sizes : list of int
        Jumlah neuron di setiap layer termasuk input dan output.
        Contoh: [10, 16, 8, 1] → input=10, hidden1=16, hidden2=8, output=1
    learning_rate : float
        Laju pembelajaran untuk update bobot (default: 0.005)
    lambda_reg : float
        Koefisien L2 regularization untuk mencegah overfitting (default: 0.001)
    random_state : int
        Seed untuk reprodusibilitas hasil (default: 42)
    """

    def __init__(self, layer_sizes, learning_rate=0.005, lambda_reg=0.001, random_state=42):
        self.layer_sizes   = layer_sizes
        self.lr            = learning_rate
        self.lambda_reg    = lambda_reg
        self.random_state  = random_state
        self.weights       = []
        self.biases        = []
        self.history       = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': []
        }
        self._init_weights()

    def _init_weights(self):
        """
        He Initialization: W ~ N(0, sqrt(2 / n_in))
        Direkomendasikan untuk jaringan dengan aktivasi ReLU karena
        mempertahankan variansi gradien selama backpropagation.
        """
        np.random.seed(self.random_state)
        for i in range(len(self.layer_sizes) - 1):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    # -----------------------------------------------------------------------
    # Fungsi Aktivasi
    # -----------------------------------------------------------------------

    def _relu(self, z):
        """ReLU: max(0, z)"""
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        """Turunan ReLU: 1 jika z > 0, else 0"""
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        """
        Sigmoid: 1 / (1 + e^(-z))
        Di-clip untuk menghindari overflow numerik.
        """
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    # -----------------------------------------------------------------------
    # Forward Propagation
    # -----------------------------------------------------------------------

    def _forward(self, X):
        """
        Propagasi maju (forward pass) melalui semua layer.

        Menyimpan nilai pre-aktivasi (Z) dan aktivasi (A) untuk
        digunakan dalam backpropagation.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        A : ndarray, shape (n_samples, 1)
            Output probabilitas dari layer terakhir (Sigmoid).
        """
        self._activations = [X]
        self._pre_activations = []
        A = X
        # Hidden layers: aktivasi ReLU
        for i in range(len(self.weights) - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self._pre_activations.append(Z)
            A = self._relu(Z)
            self._activations.append(A)
        # Output layer: aktivasi Sigmoid
        Z = A @ self.weights[-1] + self.biases[-1]
        self._pre_activations.append(Z)
        A = self._sigmoid(Z)
        self._activations.append(A)
        return A

    # -----------------------------------------------------------------------
    # Loss Function
    # -----------------------------------------------------------------------

    def _compute_loss(self, y_true, y_pred, class_weights=None):
        """
        Binary Cross-Entropy Loss dengan opsional class weighting dan L2 regularization.

        BCE = -1/m * sum(w * [y * log(p) + (1-y) * log(1-p)])
        L2  = (lambda / 2m) * sum(W^2)
        Total Loss = BCE + L2

        Parameters
        ----------
        y_true : ndarray, shape (n_samples,)
        y_pred : ndarray, shape (n_samples, 1) atau (n_samples,)
        class_weights : dict {0: w0, 1: w1}, optional

        Returns
        -------
        float : nilai total loss
        """
        eps    = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = y_true.reshape(-1, 1)
        m      = len(y_true)

        if class_weights is not None:
            w = np.where(y_true == 1, class_weights[1], class_weights[0])
            bce = -np.mean(w * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
        else:
            bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 Regularization: penalti bobot besar untuk mencegah overfitting
        l2_reg = sum(np.sum(W ** 2) for W in self.weights)
        return bce + (self.lambda_reg / (2 * m)) * l2_reg

    # -----------------------------------------------------------------------
    # Backpropagation
    # -----------------------------------------------------------------------

    def _backward(self, X, y_true, class_weights=None):
        """
        Backpropagation: menghitung gradien dan memperbarui bobot.

        Menggunakan chain rule untuk menghitung gradien dL/dW dan dL/db
        di setiap layer, kemudian memperbarui dengan gradient descent.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        y_true : ndarray, shape (n_samples,)
        class_weights : dict, optional
        """
        m      = X.shape[0]
        y_true = y_true.reshape(-1, 1)

        # Gradien dari output layer (Sigmoid + BCE)
        # dL/dA_out = A_out - y  (untuk BCE dengan Sigmoid)
        dA = self._activations[-1] - y_true
        if class_weights is not None:
            w  = np.where(y_true == 1, class_weights[1], class_weights[0])
            dA = dA * w

        # Akumulasi gradien tiap layer (dari output ke input)
        grad_W = []
        grad_b = []
        for i in range(len(self.weights) - 1, -1, -1):
            dW = (self._activations[i].T @ dA) / m + (self.lambda_reg / m) * self.weights[i]
            db = np.mean(dA, axis=0, keepdims=True)
            grad_W.insert(0, dW)
            grad_b.insert(0, db)
            # Propagasi gradien ke layer sebelumnya (lewat ReLU)
            if i > 0:
                dA = (dA @ self.weights[i].T) * self._relu_derivative(self._pre_activations[i - 1])

        # Update bobot: W = W - lr * dW
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grad_W[i]
            self.biases[i]  -= self.lr * grad_b[i]

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=500, batch_size=32, class_weights=None,
            early_stopping_patience=40, verbose=True):
        """
        Melatih model ANN menggunakan mini-batch SGD.

        Parameters
        ----------
        X_train, y_train : data latih
        X_val, y_val     : data validasi untuk early stopping
        epochs           : maks epoch pelatihan
        batch_size       : ukuran mini-batch
        class_weights    : bobot per kelas {0: w0, 1: w1}
        early_stopping_patience : jumlah epoch tanpa perbaikan sebelum berhenti
        verbose          : tampilkan progress setiap 50 epoch
        """
        m               = X_train.shape[0]
        best_val_loss   = np.inf
        patience_count  = 0
        best_weights    = [w.copy() for w in self.weights]
        best_biases     = [b.copy() for b in self.biases]

        if verbose:
            print(f"{'='*60}")
            print(f"  Memulai pelatihan ANN")
            print(f"  Arsitektur : {self.layer_sizes}")
            print(f"  Learning rate: {self.lr} | L2 lambda: {self.lambda_reg}")
            print(f"  Batch size : {batch_size} | Max epochs: {epochs}")
            print(f"{'='*60}")

        for epoch in range(epochs):
            # Shuffle data setiap epoch untuk menghindari bias urutan
            idx     = np.random.permutation(m)
            X_shuf  = X_train[idx]
            y_shuf  = y_train[idx]

            # Mini-batch gradient descent
            for start in range(0, m, batch_size):
                end = start + batch_size
                Xb  = X_shuf[start:end]
                yb  = y_shuf[start:end]
                self._forward(Xb)
                self._backward(Xb, yb, class_weights)

            # Hitung metrik epoch ini
            train_proba = self.predict_proba(X_train)
            train_loss  = self._compute_loss(y_train, train_proba, class_weights)
            train_pred  = (train_proba >= 0.5).astype(int)
            train_acc   = accuracy_score(y_train, train_pred)

            val_proba   = self.predict_proba(X_val)
            val_loss    = self._compute_loss(y_val, val_proba, class_weights)
            val_pred    = (val_proba >= 0.5).astype(int)
            val_acc     = accuracy_score(y_val, val_pred)

            self.history['loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['accuracy'].append(float(train_acc))
            self.history['val_accuracy'].append(float(val_acc))

            # Early stopping: simpan bobot terbaik berdasarkan val_loss
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = [w.copy() for w in self.weights]
                best_biases    = [b.copy() for b in self.biases]
            else:
                patience_count += 1
                if patience_count >= early_stopping_patience:
                    if verbose:
                        print(f"\n  [Early Stopping] Epoch {epoch + 1} | "
                              f"Val Loss tidak membaik selama {early_stopping_patience} epoch.")
                    break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:>4}/{epochs} | "
                      f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Restore bobot terbaik
        self.weights = best_weights
        self.biases  = best_biases
        self.best_val_loss = best_val_loss
        if verbose:
            print(f"\n  Pelatihan selesai. Best Val Loss: {best_val_loss:.4f}")
            print(f"{'='*60}\n")
        return self

    # -----------------------------------------------------------------------
    # Prediksi
    # -----------------------------------------------------------------------

    def predict_proba(self, X):
        """
        Kembalikan probabilitas kelas positif (anomali).

        Returns
        -------
        ndarray, shape (n_samples,) — probabilitas antara 0 dan 1
        """
        return self._forward(X).flatten()

    def predict(self, X, threshold=0.5):
        """
        Kembalikan label biner (0=Normal, 1=Anomali) berdasarkan threshold.

        Parameters
        ----------
        threshold : float
            Batas probabilitas untuk klasifikasi positif (default: 0.5)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def summary(self):
        """Tampilkan ringkasan arsitektur model."""
        print("=" * 50)
        print("  RINGKASAN ARSITEKTUR ANN")
        print("=" * 50)
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            params = W.size + b.size
            total_params += params
            layer_type = "Hidden" if i < len(self.weights) - 1 else "Output"
            activ      = "ReLU"   if i < len(self.weights) - 1 else "Sigmoid"
            print(f"  Layer {i+1} ({layer_type:6s}): {self.layer_sizes[i]:>3} → "
                  f"{self.layer_sizes[i+1]:>3} | Aktivasi: {activ:7s} | "
                  f"Parameter: {params}")
        print(f"  {'─'*44}")
        print(f"  Total parameter: {total_params}")
        print("=" * 50)


# ===========================================================================
# 2. FUNGSI UTILITAS
# ===========================================================================

def find_optimal_threshold(model, X_val, y_val, thresholds=None):
    """
    Mencari threshold optimal berdasarkan F1-Score di data validasi.

    Parameters
    ----------
    model      : ArtificialNeuralNetwork yang sudah dilatih
    X_val      : data validasi (sudah di-scale)
    y_val      : label validasi
    thresholds : array threshold yang dicoba (default: 0.30 – 0.70)

    Returns
    -------
    best_threshold : float
    best_f1        : float
    results_df     : DataFrame dengan semua hasil threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.71, 0.01)

    y_proba  = model.predict_proba(X_val)
    records  = []
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p     = precision_score(y_val, preds, zero_division=0)
        r     = recall_score(y_val, preds, zero_division=0)
        f1    = f1_score(y_val, preds, zero_division=0)
        acc   = accuracy_score(y_val, preds)
        records.append({'threshold': round(t, 2), 'precision': p,
                        'recall': r, 'f1': f1, 'accuracy': acc})
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1, pd.DataFrame(records)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluasi lengkap model pada data uji.

    Returns
    -------
    dict : semua metrik evaluasi
    """
    y_pred  = model.predict(X_test, threshold=threshold)
    y_proba = model.predict_proba(X_test)
    cm      = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc_roc':   roc_auc_score(y_test, y_proba),
        'confusion_matrix': cm,
        'y_pred':    y_pred,
        'y_proba':   y_proba
    }

    # TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    metrics.update({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
    return metrics


def print_evaluation_report(metrics, threshold):
    """Cetak laporan evaluasi yang terformat."""
    print("=" * 60)
    print("  LAPORAN EVALUASI MODEL ANN — DATA UJI")
    print("=" * 60)
    print(f"  Threshold optimal   : {threshold:.2f}")
    print(f"  {'─'*56}")
    print(f"  Accuracy            : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision           : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  Recall (Sensitivity): {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score            : {metrics['f1']:.4f}  ({metrics['f1']*100:.2f}%)")
    print(f"  AUC-ROC             : {metrics['auc_roc']:.4f}  ({metrics['auc_roc']*100:.2f}%)")
    print(f"  {'─'*56}")
    print(f"\n  Confusion Matrix:")
    print(f"                   Pred: Normal  Pred: Anomali")
    print(f"  Aktual: Normal      {metrics['TN']:>6}         {metrics['FP']:>6}")
    print(f"  Aktual: Anomali     {metrics['FN']:>6}         {metrics['TP']:>6}")
    print()
    print("  Classification Report:")
    print("  " + "-" * 56)
    from sklearn.metrics import classification_report
    report = classification_report(
        [0]*int(metrics['TN']+metrics['FP']) + [1]*int(metrics['FN']+metrics['TP']),
        [0]*int(metrics['TN']) + [1]*int(metrics['FP']) +
        [0]*int(metrics['FN']) + [1]*int(metrics['TP']),
        target_names=['Normal (0)', 'Anomali (1)']
    )
    for line in report.strip().split('\n'):
        print("  " + line)
    print("=" * 60)


# ===========================================================================
# 3. VISUALISASI
# ===========================================================================

def plot_all(model, metrics, X_test, y_test, dataset_info, save_path=None):
    """
    Membuat dashboard visualisasi lengkap:
    1. Distribusi kelas dataset
    2. Training & Validation Loss
    3. Training & Validation Accuracy
    4. Confusion Matrix (heatmap)
    5. Kurva ROC
    6. Distribusi probabilitas prediksi
    """
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Dashboard: Deteksi Anomali Penimbunan LPG 3 Kg\n"
        "Model Artificial Neural Network (Pure NumPy)",
        fontsize=14, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors_main = {'normal': '#378ADD', 'anomali': '#E24B4A'}

    # --- 1. Distribusi Kelas ---
    ax1 = fig.add_subplot(gs[0, 0])
    labels_dist  = ['Normal (0)', 'Anomali (1)']
    counts       = [dataset_info['normal'], dataset_info['anomali']]
    bars         = ax1.bar(labels_dist, counts,
                           color=[colors_main['normal'], colors_main['anomali']],
                           edgecolor='white', linewidth=0.8, width=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f'{count}\n({count/dataset_info["total"]*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_title('Distribusi Kelas Dataset', fontsize=11, pad=10)
    ax1.set_ylabel('Jumlah Data')
    ax1.set_ylim(0, max(counts) * 1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)

    # --- 2. Training & Validation Loss ---
    ax2 = fig.add_subplot(gs[0, 1])
    epochs_range = range(1, len(model.history['loss']) + 1)
    ax2.plot(epochs_range, model.history['loss'],
             color='#185FA5', linewidth=2, label='Train Loss')
    ax2.plot(epochs_range, model.history['val_loss'],
             color='#D85A30', linewidth=2, linestyle='--', label='Val Loss')
    ax2.set_title('Training & Validation Loss', fontsize=11, pad=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)

    # --- 3. Training & Validation Accuracy ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_range, [v * 100 for v in model.history['accuracy']],
             color='#1D9E75', linewidth=2, label='Train Acc')
    ax3.plot(epochs_range, [v * 100 for v in model.history['val_accuracy']],
             color='#BA7517', linewidth=2, linestyle='--', label='Val Acc')
    ax3.set_title('Training & Validation Accuracy', fontsize=11, pad=10)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(60, 102)
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(alpha=0.3)

    # --- 4. Confusion Matrix ---
    ax4 = fig.add_subplot(gs[1, 0])
    cm = metrics['confusion_matrix']
    im = ax4.imshow(cm, interpolation='nearest',
                    cmap=plt.cm.Blues, aspect='auto')
    ax4.set_title('Confusion Matrix', fontsize=11, pad=10)
    tick_marks = [0, 1]
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(['Normal (0)', 'Anomali (1)'], fontsize=9)
    ax4.set_yticklabels(['Normal (0)', 'Anomali (1)'], fontsize=9)
    ax4.set_xlabel('Prediksi', fontsize=10)
    ax4.set_ylabel('Aktual', fontsize=10)
    cm_labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax4.text(j, i, f'{cm[i, j]}\n({cm_labels[i][j]})',
                     ha='center', va='center',
                     color=color, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # --- 5. Kurva ROC ---
    ax5 = fig.add_subplot(gs[1, 1])
    fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
    auc_val     = metrics['auc_roc']
    ax5.plot(fpr, tpr, color='#185FA5', linewidth=2.5,
             label=f'ANN (AUC = {auc_val:.4f})')
    ax5.plot([0, 1], [0, 1], color='#888780',
             linewidth=1.5, linestyle='--', label='Random')
    ax5.fill_between(fpr, tpr, alpha=0.08, color='#185FA5')
    ax5.set_title('Kurva ROC', fontsize=11, pad=10)
    ax5.set_xlabel('False Positive Rate (FPR)')
    ax5.set_ylabel('True Positive Rate (TPR)')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1.02])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(alpha=0.3)

    # --- 6. Distribusi Probabilitas Prediksi ---
    ax6 = fig.add_subplot(gs[1, 2])
    proba_normal  = metrics['y_proba'][y_test == 0]
    proba_anomali = metrics['y_proba'][y_test == 1]
    ax6.hist(proba_normal, bins=20, alpha=0.7, color=colors_main['normal'],
             label=f'Normal (n={len(proba_normal)})', edgecolor='white')
    ax6.hist(proba_anomali, bins=20, alpha=0.7, color=colors_main['anomali'],
             label=f'Anomali (n={len(proba_anomali)})', edgecolor='white')
    ax6.axvline(x=0.49, color='black', linestyle='--',
                linewidth=1.5, label='Threshold = 0.49')
    ax6.set_title('Distribusi Probabilitas Prediksi', fontsize=11, pad=10)
    ax6.set_xlabel('Probabilitas Prediksi')
    ax6.set_ylabel('Frekuensi')
    ax6.legend(fontsize=9)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [INFO] Grafik disimpan ke: {save_path}")
    plt.show()


# ===========================================================================
# 4. MAIN PIPELINE
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("  DETEKSI ANOMALI PENIMBUNAN LPG 3 KG")
    print("  Model: Artificial Neural Network (Pure NumPy)")
    print("=" * 60 + "\n")

    # -----------------------------------------------------------------------
    # A. LOAD DATASET
    # -----------------------------------------------------------------------
    print("[1] Memuat dataset...")
    # Ganti path sesuai lokasi file CSV Anda
    DATA_PATH = "dataset_lpg_labeled_final_v2.csv"
    df = pd.read_csv(DATA_PATH, sep=';')

    print(f"    Jumlah data   : {len(df)} baris")
    print(f"    Jumlah fitur  : {df.shape[1] - 1} kolom (selain label)")
    print(f"    Kolom dataset : {df.columns.tolist()}\n")

    feature_names = [
        'kategori_enc',          # Kategori pembeli (0=non-UMKM, 1=UMKM)
        'frekuensi_beli',        # Frekuensi transaksi per 4 bulan
        'total_tabung',          # Total tabung yang dibeli
        'selalu_beli_maksimum',  # Selalu membeli kuota maksimum (0/1)
        'beli_awal_bulan',       # Aktif di awal bulan (0/1)
        'beli_tengah_bulan',     # Aktif di tengah bulan (0/1)
        'beli_akhir_bulan',      # Aktif di akhir bulan (0/1)
        'rasio_vs_rata_kategori',# Rasio pembelian vs rata-rata kategori
        'bulan_aktif_dari_4',    # Jumlah bulan aktif dari 4 periode
        'tidak_aktif_bulan_ini'  # Flag tidak aktif bulan ini (0/1)
    ]

    X = df[feature_names].values.astype(float)
    y = df['label'].values.astype(int)

    dataset_info = {
        'total':   len(df),
        'normal':  int((y == 0).sum()),
        'anomali': int((y == 1).sum())
    }
    print(f"    Distribusi label:")
    print(f"      Normal   (0): {dataset_info['normal']:>4} data "
          f"({dataset_info['normal']/dataset_info['total']*100:.1f}%)")
    print(f"      Anomali  (1): {dataset_info['anomali']:>4} data "
          f"({dataset_info['anomali']/dataset_info['total']*100:.1f}%)")
    print(f"      Rasio imbalance: 1 : {dataset_info['normal']/dataset_info['anomali']:.2f}\n")

    # -----------------------------------------------------------------------
    # B. SPLIT DATA (Train 68% / Val 12% / Test 20%)
    # -----------------------------------------------------------------------
    print("[2] Membagi dataset...")
    # Split pertama: pisahkan test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Split kedua: dari sisa, pisahkan validation (15% → ~12% dari total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15, random_state=42, stratify=y_train_val
    )

    print(f"    Train : {X_train.shape[0]} data "
          f"(Normal={int((y_train==0).sum())}, Anomali={int((y_train==1).sum())})")
    print(f"    Val   : {X_val.shape[0]} data "
          f"(Normal={int((y_val==0).sum())},  Anomali={int((y_val==1).sum())})")
    print(f"    Test  : {X_test.shape[0]} data "
          f"(Normal={int((y_test==0).sum())},  Anomali={int((y_test==1).sum())})\n")

    # -----------------------------------------------------------------------
    # C. PREPROCESSING — STANDARDISASI FITUR
    # -----------------------------------------------------------------------
    print("[3] Preprocessing — StandardScaler...")
    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)   # fit hanya pada train
    X_val_sc     = scaler.transform(X_val)          # transform val
    X_test_sc    = scaler.transform(X_test)         # transform test

    print(f"    Mean fitur (train): {np.round(scaler.mean_, 3)}")
    print(f"    Std  fitur (train): {np.round(scaler.scale_, 3)}\n")

    # -----------------------------------------------------------------------
    # D. CLASS WEIGHTING — Menangani Imbalance Data
    # -----------------------------------------------------------------------
    print("[4] Menghitung class weights untuk imbalance data...")
    n_total = len(y_train)
    n_neg   = (y_train == 0).sum()
    n_pos   = (y_train == 1).sum()
    # Rumus: w_c = n_total / (n_classes * n_c)
    w0 = n_total / (2 * n_neg)   # bobot kelas Normal
    w1 = n_total / (2 * n_pos)   # bobot kelas Anomali
    class_weights = {0: w0, 1: w1}

    print(f"    Bobot Normal  (0): {w0:.4f}")
    print(f"    Bobot Anomali (1): {w1:.4f}")
    print(f"    → Kelas anomali mendapat bobot {w1/w0:.1f}x lebih besar\n")

    # -----------------------------------------------------------------------
    # E. BANGUN & LATIH MODEL ANN
    # -----------------------------------------------------------------------
    print("[5] Membangun arsitektur ANN...")

    # Arsitektur: 10 input → 16 (ReLU) → 8 (ReLU) → 1 (Sigmoid)
    model = ArtificialNeuralNetwork(
        layer_sizes   = [10, 16, 8, 1],
        learning_rate = 0.005,
        lambda_reg    = 0.001,
        random_state  = 42
    )
    model.summary()

    print("[6] Melatih model ANN...\n")
    model.fit(
        X_train_sc, y_train,
        X_val_sc,   y_val,
        epochs                   = 500,
        batch_size               = 32,
        class_weights            = class_weights,
        early_stopping_patience  = 40,
        verbose                  = True
    )

    # -----------------------------------------------------------------------
    # F. OPTIMASI THRESHOLD
    # -----------------------------------------------------------------------
    print("[7] Mencari threshold optimal di data validasi...")
    best_threshold, best_f1, threshold_df = find_optimal_threshold(
        model, X_val_sc, y_val
    )
    print(f"    Threshold terbaik : {best_threshold:.2f}")
    print(f"    F1-Score terbaik  : {best_f1:.4f}")
    print(f"\n    5 Threshold teratas:")
    top5 = threshold_df.sort_values('f1', ascending=False).head(5)
    print(top5[['threshold', 'precision', 'recall', 'f1', 'accuracy']]
          .to_string(index=False, float_format='{:.4f}'.format))
    print()

    # -----------------------------------------------------------------------
    # G. EVALUASI MODEL
    # -----------------------------------------------------------------------
    print("[8] Evaluasi model pada data uji...")
    metrics = evaluate_model(model, X_test_sc, y_test, threshold=best_threshold)
    print_evaluation_report(metrics, best_threshold)

    # -----------------------------------------------------------------------
    # H. VISUALISASI DASHBOARD
    # -----------------------------------------------------------------------
    print("\n[9] Membuat visualisasi dashboard...")
    plot_all(
        model        = model,
        metrics      = metrics,
        X_test       = X_test_sc,
        y_test       = y_test,
        dataset_info = dataset_info,
        save_path    = "dashboard_ann_lpg.png"
    )

    # -----------------------------------------------------------------------
    # I. CONTOH PREDIKSI DATA BARU
    # -----------------------------------------------------------------------
    print("\n[10] Contoh prediksi data pembeli baru:")
    print("=" * 60)

    # Contoh data baru (sesuai urutan fitur)
    # Format: [kategori_enc, frekuensi_beli, total_tabung, selalu_beli_maksimum,
    #          beli_awal_bulan, beli_tengah_bulan, beli_akhir_bulan,
    #          rasio_vs_rata_kategori, bulan_aktif_dari_4, tidak_aktif_bulan_ini]
    contoh_data = np.array([
        # Diduga normal: UMKM, beli 2x, 3 tabung, tidak selalu maks
        [1, 2, 3, 0, 1, 0, 1, 0.75, 3, 0],
        # Diduga anomali: non-UMKM, beli 4x, 4 tabung, SELALU maks,
        #                 aktif semua periode, rasio sangat tinggi
        [0, 4, 4, 1, 1, 1, 1, 1.58, 4, 0],
        # Tidak aktif bulan ini
        [0, 0, 0, 0, 0, 0, 0, 0.0,  2, 1],
    ])

    contoh_labels = [
        "Pembeli A (UMKM, pembelian wajar)",
        "Pembeli B (Non-UMKM, indikasi penimbunan)",
        "Pembeli C (tidak aktif bulan ini)"
    ]

    contoh_scaled = scaler.transform(contoh_data)
    probas        = model.predict_proba(contoh_scaled)
    prediksi      = model.predict(contoh_scaled, threshold=best_threshold)

    for label, proba, pred in zip(contoh_labels, probas, prediksi):
        status = "⚠  ANOMALI (Indikasi Penimbunan)" if pred == 1 else "✓  NORMAL"
        print(f"  {label}")
        print(f"    Probabilitas anomali : {proba:.4f} ({proba*100:.2f}%)")
        print(f"    Klasifikasi          : {status}")
        print()

    print("=" * 60)
    print("  SELESAI — Model ANN berhasil dilatih dan dievaluasi.")
    print("=" * 60)

    return model, scaler, metrics, best_threshold


# ===========================================================================
# 5. ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    model, scaler, metrics, threshold = main()
