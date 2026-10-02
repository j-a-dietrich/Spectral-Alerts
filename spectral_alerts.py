import base64
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from matchms.exporting import save_as_json
from matchms.importing import load_from_json, load_from_mgf, load_from_mzml
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB, motifDB2motifs
from rdkit import Chem
from rdkit.Chem import Draw

from matching_utils import find_matches  # must sit next to this file

FRAG_TOL = 0.0   # m/z tolerance for fragments (after rounding to 2 decimals)
LOSS_TOL = 0.0   # m/z tolerance for neutral losses

st.set_page_config(page_title="Spectral Alerts", layout="wide")
st.title("🧪 Spectral Alerts Viewer")

# ----------------------------------------------------------------------------
# Sidebar: uploads and settings
# ----------------------------------------------------------------------------
st.sidebar.header("Upload Files")
uploaded_alerts = st.sidebar.file_uploader("Upload Spectral Alerts", type=["json"])
uploaded_spectra = st.sidebar.file_uploader(
    "Upload Sample Spectra", type=["mgf", "mzml", "json"], accept_multiple_files=True
)

st.sidebar.header("Settings")
max_cpu = os.cpu_count() or 1
n_workers = st.sidebar.number_input(
    f"CPU cores for screening (1-{max_cpu})", min_value=1, max_value=max_cpu, value=max_cpu
)
make_plots = st.sidebar.checkbox(
    "Draw spectrum plots (slower with many matches)", value=True
)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
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
            spectral_alert.set("matching_score", motifDB_grouped["matching_score"].iloc[i])
            spectral_alerts_extended.append(spectral_alert)
        return spectral_alerts_extended

    return spectral_alerts


def process_spectra(files):
    spectra = []
    for f in (files if isinstance(files, list) else [files]):
        ext = f.name.lower().split(".")[-1]

        if ext == "json":
            specs = load_from_json(BytesIO(f.read()))
        elif ext == "mgf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mgf") as tmp:
                tmp.write(f.read())
                tmp.flush()
                path = tmp.name
            specs = load_from_mgf(path)
        elif ext == "mzml":
            specs = load_from_mzml(BytesIO(f.read()))
        else:
            continue

        spectra.extend(specs)
    return spectra


# ----------------------------------------------------------------------------
# Helpers for images
# ----------------------------------------------------------------------------
def png_to_data_uri(png_bytes):
    """st.dataframe can show images given as 'data:image/png;base64,...' strings."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def mol_to_data_uri(smiles, size=(150, 150)):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    buffered = BytesIO()
    Draw.MolToImage(mol, size=size).save(buffered, format="PNG")
    return png_to_data_uri(buffered.getvalue())


def rounded_mz(peaks_or_losses):
    """Rounded m/z array of a matchms Fragments object (empty array if there is none)."""
    if peaks_or_losses is None:
        return np.array([], dtype=np.float64)
    return np.round(np.asarray(peaks_or_losses.mz, dtype=np.float64), 2)


def spectrum_match_uri(sample, alert, tol=FRAG_TOL):
    """
    Mirror plot: sample spectrum on top (red = fragment also in the alert, grey = other),
    alert spectrum at the bottom (blue).
    """
    s_mz, s_int = sample.peaks.mz, sample.peaks.intensities
    a_mz, a_int = alert.peaks.mz, alert.peaks.intensities

    if len(a_mz):
        diffs = np.abs(np.round(s_mz, 2)[:, None] - np.round(a_mz, 2)[None, :])
        matched = (diffs <= tol).any(axis=1)
    else:
        matched = np.zeros(len(s_mz), dtype=bool)

    s_norm = s_int / max(s_int.max(), 1e-12) if len(s_int) else s_int
    a_norm = a_int / max(a_int.max(), 1e-12) if len(a_int) else a_int

    # Figure (not pyplot) so no global state / memory leaks
    fig = Figure(figsize=(3.4, 1.7), dpi=80)
    ax = fig.subplots()
    ax.vlines(s_mz[~matched], 0, s_norm[~matched], color="black", linewidth=1)
    ax.vlines(s_mz[matched], 0, s_norm[matched], color="red", linewidth=1.5)
    #ax.vlines(a_mz, 0, -a_norm, color="steelblue", linewidth=1.5) # makes a mirror plot
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("m/z", fontsize=7, labelpad=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    return png_to_data_uri(buf.getvalue())


def extract_retention_time(r):
    """Retention time in minutes; NaN if unknown (keeps the column numeric and sortable)."""
    rt = r.get("retention_time")
    if not rt:
        try:
            rt = r.get("scan_start_time")[0]
        except TypeError:
            rt = np.nan
    else:
        rt = rt / 60.0
    return rt


# ----------------------------------------------------------------------------
# Screening
# ----------------------------------------------------------------------------
def run_matching(query_spectra, ref_spectra, n_workers):
    """Returns a list of (sample_index, alert_index) pairs, using several cores if requested."""
    # Reduce everything to small numpy arrays first: cheap to send to worker processes
    alerts = [(rounded_mz(q.peaks), rounded_mz(q.losses)) for q in query_spectra]
    jobs = [(i, rounded_mz(r.peaks), rounded_mz(r.losses)) for i, r in enumerate(ref_spectra)]

    worker = partial(find_matches, alerts=alerts, frag_tol=FRAG_TOL, loss_tol=LOSS_TOL)

    if n_workers > 1 and len(jobs) >= 200:  # multiprocessing only pays off for larger inputs
        # Several chunks per worker so that the work is balanced
        chunk_size = max(1, -(-len(jobs) // (n_workers * 4)))  # ceiling division
        chunks = [jobs[k:k + chunk_size] for k in range(0, len(jobs), chunk_size)]
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(worker, chunks))  # order is preserved
            return [hit for chunk_hits in results for hit in chunk_hits]
        except Exception as e:  # e.g. multiprocessing not available in this environment
            st.warning(f"Multi-core screening failed ({e}); falling back to a single core.")

    return worker(jobs)


def screen(query_spectra, ref_spectra, n_workers, make_plots):
    hits = run_matching(query_spectra, ref_spectra, n_workers)

    # Per-alert info, computed once (not once per match)
    alert_info = []
    for q in query_spectra:
        smiles = q.get("short_annotation")
        if not smiles:
            try:
                smiles = q.get("auto_annotation")[0]
            except (TypeError, IndexError):
                smiles = None
        alert_info.append({
            "id": q.get("motif_id"),
            "score": q.get("matching_score"),
            "name": q.get("scientific_name"),
            "structure": mol_to_data_uri(smiles),
        })

    matches_by_spec = defaultdict(list)
    for spec_idx, alert_idx in hits:
        matches_by_spec[spec_idx].append(alert_idx)

    rows, images = [], []
    for i, r in enumerate(ref_spectra):
        alert_idxs = matches_by_spec.get(i)

        if not alert_idxs:
            r.set("category_of_prioritization", "fragmentation-based")
            r.set("prioritized_feature", False)
            r.set("prioritization_certainty", None)
            r.set("reason_prioritized", None)
            continue

        rt = extract_retention_time(r)
        for a in alert_idxs:
            info = alert_info[a]
            rows.append({
                "Sample spec ID": i,
                "Sample Precursor": r.get("precursor_mz"),
                "Retention Time": rt,
                "Certainty": info["score"],
                "Alert Name": info["name"],
            })
            images.append({
                "Alert Structure": info["structure"],
                "Spectrum Match": spectrum_match_uri(r, query_spectra[a]) if make_plots else None,
            })

            if "category_of_prioritization" not in r.metadata:
                r.set("category_of_prioritization", "fragmentation-based")
                r.set("prioritized_feature", True)
                r.set("prioritization_certainty", info["score"])
                r.set("reason_prioritized", info["id"])
            else:
                r.set("reason_prioritized", f"{r.get('reason_prioritized')},{info['id']}")

    # Pure data (goes into the csv) and images (display only) are kept apart
    table = pd.DataFrame(rows)
    image_df = pd.DataFrame(images)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "prioritized.json")
        save_as_json(ref_spectra, path)
        with open(path) as fh:
            json_data = fh.read()

    return {"table": table, "images": image_df, "json": json_data}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if uploaded_alerts and uploaded_spectra:
    # Streamlit re-runs this whole script on every widget interaction. The expensive part is
    # therefore stored in session_state and only recomputed when the inputs change.
    run_key = (
        uploaded_alerts.name, uploaded_alerts.size,
        tuple((f.name, f.size) for f in uploaded_spectra),
        make_plots,
    )
    if st.session_state.get("run_key") != run_key:
        with st.spinner(f"Loading and screening (using up to {n_workers} core(s))..."):
            query_spectra = process_alerts(uploaded_alerts)
            ref_spectra = process_spectra(uploaded_spectra)
            st.session_state["results"] = screen(query_spectra, ref_spectra, n_workers, make_plots)
            st.session_state["run_key"] = run_key

    res = st.session_state["results"]
    table, images = res["table"], res["images"]

    st.subheader("📊 Matches Found")

    if table.empty:
        st.info("No matches found.")
    else:
        st.markdown("### 🔍 Filter Results")
        search = st.text_input("Alert Name contains", "")  # instant now: nothing is recomputed

        full = table.join(images)  # same index -> aligned row by row
        if search:
            full = full[full["Alert Name"].str.contains(search, case=False, na=False, regex=False)]

        # Click a column header to sort. The magnifier icon in the table toolbar searches all columns.
        st.dataframe(
            full,
            hide_index=True,
            use_container_width=True,
            row_height=110,
            column_config={
                "Alert Structure": st.column_config.ImageColumn("Alert Structure"),
                "Spectrum Match": st.column_config.ImageColumn(
                    "Spectrum Match", help="Top: sample (red = fragments found in the alert). Bottom: alert (blue)."
                ),
                "Sample Precursor": st.column_config.NumberColumn(format="%.4f"),
                "Retention Time": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        # csv: only the data columns, no images (all matches, independent of the filter)
        st.download_button(
            "⬇️ Download csv table",
            table.to_csv(index=False).encode("utf-8"),
            "spectral_matches.csv",
            "text/csv",
        )

    st.download_button(
        "🔽 Download JSON (PINTS24 format)",
        res["json"],
        "prioritized_results_pints24.json",
        "application/json",
    )

else:
    st.info("Please upload spectral alerts and sample data to begin with nontarget screening.")
