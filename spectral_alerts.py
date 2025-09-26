import streamlit as st
import pandas as pd
from matchms.importing import load_from_mgf, load_from_json, load_from_mzml
from matchms.similarity import CosineGreedy
from io import StringIO
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
import numpy as np
from numba import njit
from tqdm import tqdm


from matchms.filtering import default_filters
from matchms.exporting import save_as_json
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB, motifDB2motifs

st.set_page_config(page_title="Spectral Alerts", layout="wide")
st.title("🧪 Spectral Alerts Viewer")

# --- Upload
st.sidebar.header("Upload Files")
uploaded_alerts = st.sidebar.file_uploader("Upload Spectral Alerts", type=["json"])
uploaded_spectra = st.sidebar.file_uploader("Upload Sample Spectra", type=["mgf", "mzml", "json"], accept_multiple_files=True)

import tempfile
import os

@st.cache_data(show_spinner="Processing alerts...")
def process_alerts(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(file.read())
        tmp.flush()
        path = tmp.name

    _, motifDB = load_motifDB(path)
    spectral_alerts = motifDB2motifs(motifDB)
    spectral_alerts_extended = []
    if "matching_score" in motifDB.columns:
        motifDB_grouped = motifDB.groupby("scan").max()
        for i, spectral_alert in enumerate(spectral_alerts):
            spectral_alert.set("matching_score", motifDB_grouped.matching_score[i])
            spectral_alerts_extended.append(spectral_alert)

        return spectral_alerts_extended
    
    return spectral_alerts

def process_spectra(files):
    spectra = []
    for f in (files if isinstance(files, list) else [files]):
        ext = f.name.lower().split(".")[-1]
        

        if ext == "json":
            file_like = BytesIO(f.read())
            specs = load_from_json(file_like)
        elif ext == "mgf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                tmp.write(f.read())
                tmp.flush()
                path = tmp.name
            specs = load_from_mgf(path)
        elif ext == "mzml":
            file_like = BytesIO(f.read())
            specs = load_from_mzml(file_like)
        else:
            specs = []
            continue

        for s in specs:
            try:
                #spectra.append(default_filters(s))
                spectra.append(s)
            except Exception as e:
                print(f"Error filtering spectrum: {e}")

    return spectra


@st.cache_data
def mol_to_base64(mol, size=(150, 150)):
    img = Draw.MolToImage(mol, size=size)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{img_str}" />'


@njit
def is_all_within_tolerance_numba(query_vals, target_vals, tolerance):
    for q in query_vals:
        found = False
        for t in target_vals:
            if abs(q - t) <= tolerance:
                found = True
                break
        if not found:
            return False
    return True

def subset_match(q, r, frag_tolerance=0.005, loss_tolerance=0.01):
    frag_match = is_all_within_tolerance_numba(
        np.array(q.peaks.mz, dtype=np.float64),
        np.array(r.peaks.mz, dtype=np.float64),
        frag_tolerance
    )
    loss_match = is_all_within_tolerance_numba(
        np.array(q.losses.mz, dtype=np.float64),
        np.array(r.losses.mz, dtype=np.float64),
        loss_tolerance
    )
    return frag_match and loss_match 

def extract_retention_time(r):
    rt = r.get("retention_time")
    if not rt:
        rt = r.get('scan_start_time')[0]
    else:
        rt = rt / 60.0 
    return rt

# --- Run matching
if uploaded_alerts and uploaded_spectra:
    st.success("Files uploaded. Computing matches...")
    query_spectra = process_alerts(uploaded_alerts)
    print("spectra alerts done")
    ref_spectra = process_spectra(uploaded_spectra)
    print("input spectra done")

    results = []
    pints24_specs = []

    for i, r in tqdm(enumerate(ref_spectra)):
        any_match = False
        for q in query_spectra:
            spectral_alert_id = q.get("motif_id")
            matching_score = q.get("matching_score")
            name = q.get("scientific_name")

            matched = subset_match(q, r)
            if matched:
                any_match = True
                query_smiles = q.get("short_annotation")
                mol = Chem.MolFromSmiles(query_smiles) 
                mol_img = mol_to_base64(mol)


                results.append({
                    "Sample spec ID": i,
                    "Sample Precursor": r.get("precursor_mz"),
                    "Retention Time": extract_retention_time(r),
                    "Certainty": matching_score,
                    "Alert Name": name,
                    "Alert Structure": mol_img 
                })
                

                if "category_of_prioritization" not in r.metadata:
    
                    r.set("category_of_prioritization", "fragmentation-based")
                    r.set("prioritized_feature", True)
                    r.set("prioritization_certainty", matching_score)
                    r.set("reason_prioritized", spectral_alert_id)
                    
                else:
                    r.metadata["reason_prioritized"] += "," + spectral_alert_id

        
        if not any_match:
            r.set("category_of_prioritization", "fragmentation-based")
            r.set("prioritized_feature", False)
            r.set("prioritization_certainty", None)
            r.set("reason_prioritized", None)

        pints24_specs.append(r)
    print("comparison done")

    df = pd.DataFrame(results)
    st.subheader("📊 Matches Found")

    # --- Filter widgets
    st.markdown("### 🔍 Filter Results")

    filtered_df = df.copy()

    # --- Column 1: Text search (assumed to be a string column) ---
    search_col1 = st.text_input("Alert Name", "")
    if search_col1:
        filtered_df = filtered_df[filtered_df["Alert Name"].str.contains(search_col1, case=False, na=False)]

    # --- Column 2: Numeric range filter ---
    ##min_val_col2 = float(df["Sample Precursor"].min())
    ##max_val_col2 = float(df["Sample Precursor"].max())
    #range_col2 = st.slider("Range for 'Sample Precursor'", min_value=min_val_col2, max_value=max_val_col2, value=(min_val_col2, max_val_col2))
    #filtered_df = filtered_df[(filtered_df["Sample Precursor"] >= range_col2[0]) & (filtered_df["Sample Precursor"] <= range_col2[1])]

    # --- Column 2: Numeric range filter ---
    #min_val_col2 = float(df["Retention Time"].min())
    #max_val_col2 = float(df["Retention Time"].max())
    #range_col2 = st.slider("Range for 'Retention Time'", min_value=min_val_col2, max_value=max_val_col2, value=(min_val_col2, max_val_col2))
    #filtered_df = filtered_df[(filtered_df["Retention Time"] >= range_col2[0]) & (filtered_df["Retention Time"] <= range_col2[1])]

    # --- Column 2: Numeric range filter ---
    #min_val_col2 = float(df["Certainty"].min())
    #max_val_col2 = float(df["Certainty"].max())
    #range_col2 = st.slider("Range for 'Certainty'", min_value=min_val_col2, max_value=max_val_col2, value=(min_val_col2, max_val_col2))
    #filtered_df = filtered_df[(filtered_df["Certainty"] >= range_col2[0]) & (filtered_df["Certainty"] <= range_col2[1])]


    # Show filtered results
    st.write(filtered_df.to_html(escape=False), unsafe_allow_html=True)

    # Export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download csv table", csv, "spectral_matches.csv", "text/csv")

    

    # Call function before rendering the button
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmpfile:
        save_as_json(pints24_specs, tmpfile.name)
        tmpfile.seek(0)
        json_data = tmpfile.read()

    st.download_button("🔽 Download JSON (PINTS24 format)", json_data, "prioritized_results_pints24.json", "application/json")



else:
    st.info("Please upload spectral alerts and sample data to begin with nontarget screening.")
