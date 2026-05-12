# Eksperimen SML - Muhamad Iqbal Reza

Submission **Membangun Sistem Machine Learning (MSML)** - Dicoding x Microsoft Elevate.

## Kriteria 1: Eksperimen terhadap Dataset Pelatihan

Repository ini berisi eksperimen preprocessing dataset **Telco Customer Churn** untuk klasifikasi binary prediction churn pelanggan.

## Struktur Folder
```bash
Eksperimen_SML_MuhamadIqbalReza/
│
├── .github/
│   └── workflows/
│       └── preprocessing.yml          # GitHub Actions workflow
│
├── Eksperimen_MuhamadIqbalReza.ipynb # Notebook eksperimen
│
├── automate_MuhamadIqbalReza.py      # Script otomatisasi preprocessing
│
├── telco_churn_raw.csv               # Dataset mentah
├── telco_churn_clean.csv             # Dataset hasil cleaning
│
├── preprocessing/
│   ├── telco_churn_preprocessing/    # Hasil preprocessing
│   │
│   ├── X_train.csv                   # Fitur data train
│   ├── X_test.csv                    # Fitur data test
│   │
│   ├── y_train.csv                   # Target data train
│   └── y_test.csv                    # Target data test
│
└── README.md
```

## Dataset

- **Sumber**: Kaggle - Telco Customer Churn (IBM Sample Data)
- **Jumlah baris**: 7.043 pelanggan
- **Jumlah kolom**: 21 fitur
- **Target**: `Churn` (Yes/No)
- **Churn rate**: 26.54%

## Tahapan Preprocessing

1. Drop kolom identifier (`customerID`)
2. Konversi `TotalCharges` ke numerik + handle 11 hidden missing
3. Konsolidasi nilai redundant (`No internet service` → `No`)
4. Label Encoding untuk fitur biner (12 fitur)
5. One-Hot Encoding untuk fitur multi-kategori (3 fitur)
6. Feature scaling dengan `StandardScaler` (3 fitur numerik)
7. Stratified train-test split (80:20)

## Environment

- Python 3.12.7
- pandas, numpy, scikit-learn, matplotlib, seaborn