# save as privacy_similarity_app.py
import streamlit as st
import numpy as np
import pandas as pd
import io
import os
import base64
import pickle
import re
import logging
import json
from datetime import datetime

from phe import paillier
from scipy.spatial import distance
from sklearn.decomposition import PCA
import plotly.express as px

SCALING_FACTOR = 1000  # multiply floats by this before encryption
KEY_SHEET_NAME = "keys"
DATA_SHEET_NAME = "data"

# -----------------------
# Logging
# -----------------------
class PrivacyLogger:
    def __init__(self, log_file='privacy_search.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        self.logger = logging.getLogger(__name__)

    def log_search_operation(self, query_vector, results, user_id=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_vector': np.asarray(query_vector).tolist(),
            'results_count': len(results),
            'user_id': user_id or 'anonymous'
        }
        self.logger.info(f"Similarity Search: {json.dumps(log_entry)}")


# -----------------------
# Utilities
# -----------------------
def generate_keys():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

class MultiLayerEncryption:
    def __init__(self, public_key):
        self.public_key = public_key

    def encrypt_row(self, numeric_values):
        """ numeric_values: iterable of floats/strings convertible to float """
        ints = [int(float(x) * SCALING_FACTOR) for x in numeric_values]
        return [self.public_key.encrypt(i) for i in ints]

def extract_cipher_int(cell_value):
    """Extract an integer from a cell that might be stored as text or number."""
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        raise ValueError("Empty ciphertext cell encountered.")
    s = str(cell_value)
    m = re.search(r'-?\d+', s)
    if not m:
        raise ValueError(f"Could not parse ciphertext int from cell: {s}")
    return int(m.group(0))

def build_encrypted_xlsx_bytes(original_df, numeric_cols, encrypted_matrix, public_key, private_key):
    """
    original_df: pandas DataFrame (all columns, text preserved)
    numeric_cols: list of numeric column names (subset of original_df.columns)
    encrypted_matrix: list of lists of ciphertext ints (as Python int or string). shape (n_rows, len(numeric_cols))
    returns: bytes of an xlsx file (two sheets: DATA_SHEET_NAME and KEY_SHEET_NAME)
    """
    # Prepare encrypted DataFrame: copy original, replace numeric columns with ciphertext strings
    df_enc = original_df.copy()
    for i, col in enumerate(numeric_cols):
        df_enc[col] = [str(row[i]) for row in encrypted_matrix]

    # Prepare keys: pickle and base64 encode a keypair object (public, private)
    keypair_obj = {"public": public_key, "private": private_key}
    key_bytes = pickle.dumps(keypair_obj)
    key_b64 = base64.b64encode(key_bytes).decode("utf-8")
    keys_df = pd.DataFrame([{"key_b64": key_b64}])

    # Write both sheets to BytesIO using openpyxl engine
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_enc.to_excel(writer, sheet_name=DATA_SHEET_NAME, index=False)
        keys_df.to_excel(writer, sheet_name=KEY_SHEET_NAME, index=False)
    output.seek(0)
    return output.read()

def parse_encrypted_xlsx_bytes(xlsx_bytes):
    """
    Read bytes of xlsx produced by build_encrypted_xlsx_bytes.
    Returns: (data_df, keypair_obj)
    """
    bio = io.BytesIO(xlsx_bytes)
    try:
        xls = pd.read_excel(bio, sheet_name=[DATA_SHEET_NAME, KEY_SHEET_NAME], dtype=str, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read uploaded .xlsx file: {e}")

    if DATA_SHEET_NAME not in xls or KEY_SHEET_NAME not in xls:
        raise ValueError("Uploaded .xlsx missing required sheets 'data' and/or 'keys'")

    data_df = xls[DATA_SHEET_NAME].astype(str)
    keys_df = xls[KEY_SHEET_NAME]

    if "key_b64" not in keys_df.columns:
        raise ValueError("Keys sheet missing 'key_b64' column")

    key_b64 = str(keys_df.loc[0, "key_b64"])
    key_bytes = base64.b64decode(key_b64)
    keypair_obj = pickle.loads(key_bytes)
    return data_df, keypair_obj

# VP-tree-like brute force (simple)
class OptimizedVPTree:
    def __init__(self, numeric_array, distance_metric='euclidean'):
        self.data = np.asarray(numeric_array, dtype=float)
        self.distance_metric = distance_metric
        self.distance_func = {
            'euclidean': distance.euclidean,
            'manhattan': distance.cityblock,
            'cosine': distance.cosine
        }.get(distance_metric, distance.euclidean)

    def search(self, query_vector, k=3):
        query_vector = np.asarray(query_vector, dtype=float)
        distances = [self.distance_func(query_vector, row) for row in self.data]
        idx = np.argsort(distances)[:k]
        return self.data[idx], np.array(distances)[idx]

# -----------------------
# Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="Privacy-Preserving Similarity Search", layout="wide")
    st.title("Privacy-Preserving Similarity Search")

    st.sidebar.header("Dataset Management")

    dataset_type = st.sidebar.radio("Select Dataset Type:", ("Raw Dataset", "Encrypted Dataset"))

    # Upload control (same as before, same place)
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls", "json"])

    # Keep session keypair for immediate session-decrypt if user encrypted in-session
    if "last_keypair" not in st.session_state:
        st.session_state["last_keypair"] = None

    if uploaded_file is None:
        st.info("Please upload a dataset to begin.")
        return

    # For logging
    privacy_logger = PrivacyLogger()

    try:
        # Branch: Raw Dataset -> encrypt, decrypt, build VP-tree from decrypted values
        if dataset_type == "Raw Dataset":
            uploaded_file.seek(0)
            filename = uploaded_file.name.lower()
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(uploaded_file, engine="openpyxl", dtype=str)
            elif filename.endswith(".json"):
                df = pd.read_json(uploaded_file, dtype=str)
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, dtype=str)

            df = df.fillna("").astype(str)
            st.subheader("Raw Dataset: ")
            st.dataframe(df)

            # Detect numeric columns robustly
            numeric_cols = []
            for col in df.columns:
                coerced = pd.to_numeric(df[col].replace("", np.nan), errors='coerce')
                non_empty = ~df[col].replace("", np.nan).isna()
                if non_empty.sum() == 0:
                    continue
                num_convertible = coerced.notna().sum()
                if num_convertible >= max(1, int(0.5 * non_empty.sum())):
                    numeric_cols.append(col)

            # Build numeric_matrix from detected numeric columns
            if len(numeric_cols) > 0:
                numeric_matrix_plain = []
                for _, row in df.iterrows():
                    vals = []
                    for c in numeric_cols:
                        try:
                            vals.append(float(row[c]))
                        except Exception:
                            vals.append(0.0)
                    numeric_matrix_plain.append(vals)
                numeric_matrix_plain = np.array(numeric_matrix_plain, dtype=float)
            else:
                numeric_matrix_plain = np.zeros((len(df), 0), dtype=float)

            # Generate key pair and encrypt numeric columns
            public_key, private_key = generate_keys()
            st.session_state["last_keypair"] = {"public": public_key, "private": private_key}
            encryptor = MultiLayerEncryption(public_key)

            encrypted_rows = []
            # encrypt each numeric row and collect ciphertext ints
            if numeric_matrix_plain.shape[1] > 0:
                for row_vals in numeric_matrix_plain:
                    enc_objs = encryptor.encrypt_row(row_vals)
                    encrypted_rows.append([enc.ciphertext() for enc in enc_objs])
            else:
                encrypted_rows = [[] for _ in range(len(df))]

            # Now --- IMPORTANT: decrypt what we just encrypted (roundtrip) and use decrypted values to build VP-tree
            decrypted_after_encrypt = []
            if len(encrypted_rows) > 0 and numeric_matrix_plain.shape[1] > 0:
                for r_idx, row in enumerate(encrypted_rows):
                    dec_row = []
                    for c_idx, ct_int in enumerate(row):
                        # wrap as EncryptedNumber and decrypt
                        enc_obj = paillier.EncryptedNumber(public_key, int(ct_int), exponent=0)
                        try:
                            dec_val = private_key.decrypt(enc_obj)
                        except Exception as e:
                            raise ValueError(f"Immediate roundtrip decryption failed at r{r_idx},c{c_idx}: {e}")
                        dec_row.append(dec_val / SCALING_FACTOR)
                    decrypted_after_encrypt.append(dec_row)
                decrypted_after_encrypt = np.array(decrypted_after_encrypt, dtype=float)
            else:
                decrypted_after_encrypt = np.zeros((len(df), 0), dtype=float)

            # Build .xlsx bytes with two sheets (data + keys) for download
            xlsx_bytes = build_encrypted_xlsx_bytes(df, numeric_cols, encrypted_rows, public_key, private_key)

            # Display encrypted numeric columns in UI (as strings)
            if len(numeric_cols) > 0:
                enc_display = pd.DataFrame(
                    {numeric_cols[i]: [str(row[i]) for row in encrypted_rows] for i in range(len(numeric_cols))}
                )
                st.subheader("Encrypted Numeric Columns") # Display only encrypted numeric columns, ciphertext integers as strings
                st.dataframe(enc_display)
            else:
                st.info("No numeric columns detected, nothing encrypted.")

            # Build VP-tree on the decrypted_after_encrypt (not the original plaintext)
            if decrypted_after_encrypt.shape[1] > 0:
                vp_tree = OptimizedVPTree(decrypted_after_encrypt, distance_metric='euclidean')
                st.success("VP-tree built from decrypted values.") # VP-tree on decrypted values after encryption roundtrip
            else:
                vp_tree = None
                st.info("No numeric features available for similarity search after roundtrip.")
            if st.sidebar.checkbox("Visualize Dataset Embedding"):
                plot_type = st.sidebar.selectbox("Choose Visualization Type:", ["Scatter Plot", "Histogram"])
                try:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(vp_tree.data)

                    if plot_type == "Scatter Plot":
                        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], title="Dataset Scatter Plot (PCA Reduced)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Scatter Plot Insights
                        - Each point represents a dataset record after dimension reduction.
                        - Points closer together imply higher similarity.
                        - Helps identify possible clusters and groupings.
                        """)

                    elif plot_type == "Histogram":
                        fig = px.histogram(x=reduced_data[:, 0], title="Dataset Histogram (First PCA Component)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Histogram Insights
                        - Shows distribution of dataset values along the primary PCA component.
                        - Identifies most common value ranges and spread.
                        - Useful to detect variance and skewness.
                        """)

                except Exception as e:
                    st.error(f"Visualization Error: {e}")
            # Download button
            st.sidebar.markdown("### Download")
            st.sidebar.download_button(
                label="Download Encrypted Dataset (.xlsx)",
                data=xlsx_bytes,
                file_name="encrypted_dataset_with_keys.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Query UI (operate on vp_tree built from decrypted_after_encrypt)
            st.sidebar.header("Query")
            if decrypted_after_encrypt.shape[1] > 0:
                q_placeholder = f"Enter {decrypted_after_encrypt.shape[1]} numeric values"
                query_input = st.sidebar.text_input("Enter Patient ID,Age,Weight,Blood Pressure,Cholestrol,Disease Risk Score (comma-separated):", placeholder=q_placeholder)
                k = st.sidebar.slider("Select number of similar results (k):", 1, max(1, decrypted_after_encrypt.shape[0]), 3)
                if query_input:
                    try:
                        qvec = np.array([float(x) for x in query_input.split(",")])
                    except Exception:
                        st.error("Invalid query vector: ensure comma-separated numeric values.")
                        qvec = None
                    if qvec is not None:
                        if qvec.size != decrypted_after_encrypt.shape[1]:
                            st.error(f"Query must have {decrypted_after_encrypt.shape[1]} numeric values.")
                        else:
                            results, dists = vp_tree.search(qvec, k)
                            st.write(f"Top {k} Similar Records (plaintext numeric):")
                            header = [
                                "Patient ID",
                                "Age",
                                "Weight",
                                "Blood Pressure",
                                "Cholesterol",
                                "Disease Risk Score"
                            ]
                            df_results = pd.DataFrame(results, columns=header)
                            df_results["distance"] = dists
                            st.dataframe(df_results)
                            privacy_logger.log_search_operation(qvec, results.tolist())
            else:
                st.sidebar.info("No numeric features available for similarity search.")

        else:
            # Encrypted Dataset branch: skip encryption, decrypt and build VP-tree
            uploaded_file.seek(0)
            raw_bytes = uploaded_file.read()

            data_df, keypair_obj = parse_encrypted_xlsx_bytes(raw_bytes)
            st.subheader("Uploaded Encrypted Dataset (raw view)")
            st.dataframe(data_df)

            # Detect encrypted columns as those where every cell yields an integer via regex
            enc_cols = []
            text_cols = []
            for col in data_df.columns:
                all_int = True
                for v in data_df[col].astype(str):
                    if v == "" or pd.isna(v):
                        all_int = False
                        break
                    try:
                        _ = extract_cipher_int(v)
                    except Exception:
                        all_int = False
                        break
                if all_int:
                    enc_cols.append(col)
                else:
                    text_cols.append(col)

            if len(enc_cols) == 0:
                st.warning("No encrypted numeric columns detected in uploaded file.")
                numeric_matrix = np.zeros((len(data_df), 0), dtype=float)
            else:
                pub = keypair_obj["public"]
                priv = keypair_obj["private"]
                decrypted_rows = []
                for r_idx, row in data_df.iterrows():
                    dec_row = []
                    for col in enc_cols:
                        try:
                            ct_int = extract_cipher_int(row[col])
                        except Exception as e:
                            raise ValueError(f"Error parsing ciphertext at row {r_idx}, col {col}: {e}")
                        enc_obj = paillier.EncryptedNumber(pub, int(ct_int), exponent=0)
                        try:
                            dec_val = priv.decrypt(enc_obj)
                        except Exception as e:
                            raise ValueError(f"Decryption failed at row {r_idx}, col {col}: {e}")
                        dec_row.append(dec_val / SCALING_FACTOR)
                    decrypted_rows.append(dec_row)
                numeric_matrix = np.array(decrypted_rows, dtype=float)
                df_decrypted = pd.DataFrame()
                for c in text_cols:
                    df_decrypted[c] = data_df[c].astype(str)
                for i, c in enumerate(enc_cols):
                    df_decrypted[c] = numeric_matrix[:, i]
                st.subheader("Decrypted Dataset ") # Show decrypted dataset (text preserved + decrypted numeric)
                st.dataframe(df_decrypted)
                # -----------------------
                # Download Decrypted Dataset (.xlsx)
                # -----------------------
                decrypted_output = io.BytesIO()

                with pd.ExcelWriter(decrypted_output, engine="openpyxl") as writer:
                    df_decrypted.to_excel(writer, index=False, sheet_name="decrypted_data")
                decrypted_output.seek(0)
                
            if numeric_matrix.shape[1] > 0:
                vp_tree = OptimizedVPTree(numeric_matrix, distance_metric='euclidean')
                st.success("VP-tree built from decrypted numeric data.")
            else:
                vp_tree = None
                st.info("No numeric features available for similarity search.")
            if st.sidebar.checkbox("Visualize Dataset Embedding"):
                plot_type = st.sidebar.selectbox("Choose Visualization Type:", ["Scatter Plot", "Histogram"])
                try:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(vp_tree.data)

                    if plot_type == "Scatter Plot":
                        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], title="Dataset Scatter Plot (PCA Reduced)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Scatter Plot Insights
                        - Each point represents a dataset record after dimension reduction.
                        - Points closer together imply higher similarity.
                        - Helps identify possible clusters and groupings.
                        """)

                    elif plot_type == "Histogram":
                        fig = px.histogram(x=reduced_data[:, 0], title="Dataset Histogram (First PCA Component)")
                        st.plotly_chart(fig)

                        st.markdown("""
                        ### 📌 Histogram Insights
                        - Shows distribution of dataset values along the primary PCA component.
                        - Identifies most common value ranges and spread.
                        - Useful to detect variance and skewness.
                        """)

                except Exception as e:
                    st.error(f"Visualization Error: {e}")
            # Re-download uploaded file if wanted
            st.sidebar.markdown("### Download")
            st.sidebar.download_button(
                label="Download Decrypted Dataset (.xlsx)",
                data=decrypted_output,
                file_name="decrypted_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Query UI (on vp_tree from decrypted numeric)
            st.sidebar.header("Query")
            if 'numeric_matrix' in locals() and numeric_matrix.shape[1] > 0:
                q_placeholder = f"Enter {numeric_matrix.shape[1]} numeric values"
                query_input = st.sidebar.text_input("Enter Patient ID,Age,Weight,Blood Pressure,Cholestrol,Disease Risk Score (comma-separated):",placeholder=q_placeholder)
                k = st.sidebar.slider("Select number of similar results (k):", 1, max(1, numeric_matrix.shape[0]), 3)
                if query_input:
                    try:
                        qvec = np.array([float(x) for x in query_input.split(",")])
                    except Exception:
                        st.error("Invalid query vector: ensure comma-separated numeric values.")
                        qvec = None
                    if qvec is not None:
                        if qvec.size != numeric_matrix.shape[1]:
                            st.error(f"Query must have {numeric_matrix.shape[1]} numeric values.")
                        else:
                            results, dists = vp_tree.search(qvec, k)
                            st.write(f"Top {k} Similar Records (plaintext numeric):")
                            header = [
                                "Patient ID",
                                "Age",
                                "Weight",
                                "Blood Pressure",
                                "Cholesterol",
                                "Disease Risk Score"
                            ]
                            df_results = pd.DataFrame(results, columns=header)
                            df_results["distance"] = dists
                            st.dataframe(df_results)
                            privacy_logger.log_search_operation(qvec, results.tolist())
            else:
                st.sidebar.info("No numeric features available for similarity search.")

    except Exception as e:
        st.error(f"Dataset Processing Error: {e}")

if __name__ == "__main__":
    main()

