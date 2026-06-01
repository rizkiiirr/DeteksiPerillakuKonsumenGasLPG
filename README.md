# Deteksi Anomali Perilaku Pembelian LPG 3 Kg Berbasis Artificial Neural Network (ANN) dan SMOTE

Repositori ini memuat kode sumber, dataset, dan dokumentasi untuk sistem kecerdasan buatan yang dirancang mendeteksi indikasi penimbunan Liquefied Petroleum Gas (LPG) 3 Kg bersubsidi di tingkat pangkalan. Proyek ini mengimplementasikan pemodelan *Machine Learning* untuk mengotomatisasi pengawasan distribusi secara objektif dan *data-driven*.

## 1. Pengenalan

Pendistribusian LPG 3 Kg bersubsidi rentan terhadap anomali perilaku pembelian, seperti penimbunan oleh oknum non-UMKM yang mengeksploitasi celah kuota. Sistem ini memecahkan masalah tersebut dengan memetakan pola transaksi pembeli menggunakan algoritma **Multilayer Perceptron (MLP)**, sebuah arsitektur *Artificial Neural Network* (ANN) *feedforward*.

Tantangan utama dalam domain pengawasan ini adalah asimetri kelas data (*imbalanced data*), di mana jumlah pembeli wajar (Normal) mendominasi secara absolut dibandingkan entitas penimbun (Anomali). Untuk mencegah *majority bias*, sistem ini diintegrasikan dengan algoritma **Synthetic Minority Over-sampling Technique (SMOTE)** yang menyintesis ruang vektor kelas minoritas secara terisolasi pada fase pelatihan silang (*cross-validation*), memastikan model mencapai sensitivitas (*Recall*) yang presisi tanpa kebocoran data (*data leakage*).

**Spesifikasi Arsitektur Model:**
* **Input Layer:** 23 Fitur Spasial (agregasi pivot transaksi 4 bulan).
* **Hidden Layers:** 64 Neuron (ReLU) dan 32 Neuron (ReLU).
* **Output Layer:** 1 Neuron (Sigmoid) untuk klasifikasi probabilitas biner.
* **Optimizer & Loss:** Adaptive Moment Estimation (Adam) dan Binary Cross-Entropy.

## 2. Prasyarat Sistem (*Dependencies*)

Untuk mengeksekusi komputasi pada repositori ini, pastikan sistem operasi Anda memiliki instalasi Python 3.8 atau yang lebih baru, serta pustaka komputasi berikut:

* `numpy` (Manipulasi aljabar linear)
* `pandas` (Pemrosesan dataset tabular)
* `scikit-learn` (Infrastruktur arsitektur MLP dan metrik evaluasi)
* `matplotlib` & `seaborn` (Visualisasi plot analitik)

Anda dapat menginstal seluruh ketergantungan tersebut menggunakan *package manager* `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn