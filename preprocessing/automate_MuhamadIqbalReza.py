"""
automate_MuhamadIqbalReza.py
============================
Script otomatisasi preprocessing dataset Telco Customer Churn.
Mengubah dataset mentah menjadi dataset siap latih untuk modelling.

Tahapan preprocessing:
    1. Load dataset
    2. Drop kolom identifier (customerID)
    3. Handle TotalCharges (konversi tipe + imputasi 11 missing)
    4. Konsolidasi nilai redundant ('No internet service' -> 'No')
    5. Label Encoding untuk fitur biner
    6. One-Hot Encoding untuk fitur multi-kategori
    7. Train-test split (stratified, 80:20)
    8. Feature scaling (StandardScaler) - fit di train, transform di test
    9. Save hasil ke folder telco_churn_preprocessing/

Usage:
    python automate_MuhamadIqbalReza.py
    python automate_MuhamadIqbalReza.py --input ../telco_churn_raw/telco_churn_raw.csv --output ../telco_churn_preprocessing
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def setup_logging():
    """Setup format logging untuk output yang informatif."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Random seed untuk reproducibility
RANDOM_SEED = 42

def load_data(input_path):
    """Load dataset dari CSV."""
    logger.info(f"Loading dataset dari: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Dataset berhasil dimuat - Shape: {df.shape}")
    return df

def drop_identifier(df):
    """Drop kolom customerID yang tidak punya nilai prediktif."""
    logger.info("Step 2: Drop kolom identifier (customerID)")
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        logger.info(f"  customerID berhasil di-drop - Shape baru: {df.shape}")
    else:
        logger.warning("  Kolom customerID tidak ditemukan, skip")
    
    return df

def handle_total_charges(df):
    """
    Konversi TotalCharges dari object ke numerik.
    Imputasi 11 hidden missing (string kosong) dengan 0 (karena tenure=0).
    """
    logger.info("Step 3: Handle kolom TotalCharges")
    
    # Konversi ke numerik (string kosong jadi NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    nan_count = df['TotalCharges'].isnull().sum()
    logger.info(f"  Hidden missing terdeteksi: {nan_count} baris")
    
    # Imputasi dengan 0 (pelanggan baru tenure=0 belum ada total tagihan)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    logger.info(f"  Imputasi selesai - TotalCharges dtype: {df['TotalCharges'].dtype}")
    
    return df

def consolidate_redundant_values(df):
    """
    Replace 'No internet service' dan 'No phone service' menjadi 'No'
    untuk mengurangi redundansi dengan fitur InternetService/PhoneService.
    """
    logger.info("Step 4: Konsolidasi nilai redundant")
    
    redundant_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
    redundant_phone = ['MultipleLines']
    
    for col in redundant_internet:
        df[col] = df[col].replace('No internet service', 'No')
    
    for col in redundant_phone:
        df[col] = df[col].replace('No phone service', 'No')
    
    logger.info(f"  Konsolidasi selesai untuk {len(redundant_internet) + len(redundant_phone)} fitur")
    return df

def encode_binary_features(df):
    """
    Encode fitur biner Yes/No -> 1/0
    Encode gender Male/Female -> 1/0
    Encode target Churn Yes/No -> 1/0
    """
    logger.info("Step 5: Encoding fitur biner")
    
    binary_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'PaperlessBilling']
    
    for col in binary_features:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    logger.info(f"  Total fitur biner di-encode: {len(binary_features) + 2}")
    return df

def one_hot_encode(df):
    """One-Hot Encoding untuk fitur multi-kategori dengan drop_first=True."""
    logger.info("Step 6: One-Hot Encoding fitur multi-kategori")
    
    multi_cat_features = ['InternetService', 'Contract', 'PaymentMethod']
    
    df = pd.get_dummies(df, columns=multi_cat_features, drop_first=True, dtype=int)
    
    logger.info(f"  One-Hot Encoding selesai - Shape baru: {df.shape}")
    return df

def split_and_scale(df, test_size=0.2, random_state=RANDOM_SEED):
    """
    Split data menjadi train-test, lalu scale fitur numerik.
    PENTING: Scaler di-fit HANYA pada train data untuk hindari data leakage.
    """
    logger.info("Step 7: Train-Test Split (stratified)")
    
    # Pisahkan fitur dan target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Stratified split untuk pertahankan rasio Churn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"  Train shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Test shape : X={X_test.shape}, y={y_test.shape}")
    logger.info(f"  Churn rate train: {y_train.mean()*100:.2f}%")
    logger.info(f"  Churn rate test : {y_test.mean()*100:.2f}%")
    
    # Step 8: Feature Scaling (fit di train only)
    logger.info("Step 8: Feature Scaling (fit di train only - no data leakage)")
    
    numeric_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    # Fit di train data, lalu transform train & test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_to_scale] = scaler.fit_transform(X_train[numeric_to_scale])
    X_test_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    
    logger.info(f"  Scaling selesai untuk {len(numeric_to_scale)} fitur numerik")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def save_outputs(X_train, X_test, y_train, y_test, output_dir):
    """Save hasil preprocessing ke folder output."""
    logger.info(f"Step 9: Save hasil preprocessing ke {output_dir}/")
    
    # Buat folder jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Save split data
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save dataset gabungan (untuk kompatibilitas dengan modelling.py)
    df_clean = pd.concat([
        pd.concat([X_train, X_test], axis=0).sort_index(),
        pd.concat([y_train, y_test], axis=0).sort_index()
    ], axis=1)
    df_clean.to_csv(os.path.join(output_dir, "telco_churn_clean.csv"), index=False)
    
    logger.info(f"  X_train.csv             - {X_train.shape}")
    logger.info(f"  X_test.csv              - {X_test.shape}")
    logger.info(f"  y_train.csv             - {y_train.shape}")
    logger.info(f"  y_test.csv              - {y_test.shape}")
    logger.info(f"  telco_churn_clean.csv   - {df_clean.shape}")

def preprocess_pipeline(input_path, output_dir, test_size=0.2, random_state=RANDOM_SEED):
    """
    Orchestrator utama yang menjalankan seluruh pipeline preprocessing.
    
    Args:
        input_path (str): Path ke dataset CSV mentah
        output_dir (str): Folder output untuk hasil preprocessing
        test_size (float): Proporsi test set (default: 0.2)
        random_state (int): Seed untuk reproducibility (default: 42)
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING TELCO CUSTOMER CHURN")
    logger.info("=" * 60)
    
    # Jalankan setiap tahap secara berurutan
    df = load_data(input_path)
    df = drop_identifier(df)
    df = handle_total_charges(df)
    df = consolidate_redundant_values(df)
    df = encode_binary_features(df)
    df = one_hot_encode(df)
    X_train, X_test, y_train, y_test = split_and_scale(df, test_size, random_state)
    save_outputs(X_train, X_test, y_train, y_test, output_dir)
    
    logger.info("=" * 60)
    logger.info("PIPELINE PREPROCESSING SELESAI - DATA SIAP UNTUK MODELLING")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset Telco Customer Churn"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../telco_churn_raw/telco_churn_raw.csv",
        help="Path ke dataset CSV mentah"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../telco_churn_preprocessing",
        help="Folder output untuk hasil preprocessing"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi test set (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_SEED,
        help="Random seed untuk reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    preprocess_pipeline(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )