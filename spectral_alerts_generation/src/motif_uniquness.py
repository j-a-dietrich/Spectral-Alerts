# here it should be checked if an optimized motif pattern is present in a large dataset with massql and then their substructure overlap score
# should be calculated and the average overlap score (0 bad; 1 perfect) will be highlighted in motifDB

# maybe this needs to be added to massql4mass2motifs
from MS2LDA.Add_On.Fingerprints.FP_annotation import annotate_motifs as calc_fingerprints
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import load_motifDB
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import motifDB2motifs
from MS2LDA.Add_On.MassQL.MassQL4MotifDB import group_ms2

from massql import msql_engine
from massql import msql_fileloading

from matchms.importing import load_from_mgf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from time import time


from rdkit import Chem
from rdkit.Chem import MACCSkeys
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def compute_maccs(smiles):
    """Compute MACCS fingerprint for a single SMILES and return as a NumPy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = MACCSkeys.GenMACCSKeys(mol)  # RDKit BitVect
        return np.array(fp)  # Convert to NumPy array
    return None  # Return None if molecule is invalid

def calculate_maccs_parallel(smiles_list, num_cores=None):
    """Compute MACCS fingerprints in parallel and return as NumPy arrays."""
    if num_cores is None:
        num_cores = max(1, cpu_count() - 1)  # Use all but 1 core

    fingerprints = Parallel(n_jobs=num_cores)(
        delayed(compute_maccs)(smi) for smi in smiles_list
    )
    
    return np.array([fp for fp in fingerprints if fp is not None])




def calculate_sos(fp1, fp2, in_order=True):
    fp1, fp2 = np.array(fp1), np.array(fp2)  # Convert to NumPy arrays if not already
    
    if not in_order:
        if fp1.sum() < fp2.sum():
            smaller_fp, bigger_fp = fp1, fp2
        else:
            smaller_fp, bigger_fp = fp2, fp1
    else:
        smaller_fp, bigger_fp = fp1, fp2

    # Use NumPy for efficient calculation
    smaller_fp_sum = smaller_fp.sum()
    fp_intersection = np.sum(np.logical_and(smaller_fp, bigger_fp))

    return fp_intersection / smaller_fp_sum if fp_intersection != 0 else 0


def color_gradient(val):
    cmap = plt.get_cmap('RdYlGn')  # Red → Yellow → Green colormap
    norm_val = val**2  # Apply a non-linear scaling to exaggerate differences
    color = cmap(norm_val)  # Get RGBA color from colormap
    rgb = tuple(int(c * 255) for c in color[:3])  # Convert to RGB
    
    return f'background-color: rgb{rgb}'


def motif2query(motif):
    peaks = motif.peaks.mz
    losses = motif.losses.mz

    query = "QUERY scaninfo(MS2DATA) WHERE"
    if peaks.any():
        for peak in peaks:
            if query[-3:] == "005":
                query += " AND"
            query += f" MS2MZ={peak}:TOLERANCEMZ=0.005"

    if losses.any():
        for loss in losses:
            if query[-3:] == "005" or query[-2:] == "01":
                query += " AND"
            query += f" MS2NL={loss}:TOLERANCEMZ=0.01"
            
    return query


def retrieve_massql_smiles(query_results, massql_spectra):
    smiles = []
    for scan_number in query_results.scan:
        smi = massql_spectra[scan_number].get("smiles")
        smiles.append(smi)
    return smiles


def calc_uniqueness_score(motifDB):
    # motifset or motifDB as input
    # manual set input parameters
    massql_db = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP2\Spectral_Alerts\3_m2m_characterization\2_m2mUniqueness\massql_ref_library.mgf"
   
    ms1_spec = pd.read_feather(r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP2\Spectral_Alerts\3_m2m_characterization\2_m2mUniqueness\ms1_df.feather")
    ms2_spec = pd.read_feather(r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\WP2\Spectral_Alerts\3_m2m_characterization\2_m2mUniqueness\ms2_df.feather")
  
    #--------------

    massql_spectra = list(tqdm(load_from_mgf(massql_db), desc="Loading spectra"))
    
    #s1_spec, ms2_spec = msql_fileloading.load_data(massql_db)
    #rint("Massql DB loaded")
   
    _, ms2_motifDB= load_motifDB(motifDB)
    grouped_ms2_motifDB = group_ms2(ms2_motifDB)
    # use optimized motifs
    motifs = motifDB2motifs(ms2_motifDB)

    queries = []
    matching_scores = []
    for motif in tqdm(motifs, desc="Querying spectra"):
        query = motif2query(motif)
        queries.append(query)
        query_results = msql_engine.process_query(query, massql_db, ms1_df=ms1_spec, ms2_df=ms2_spec)
        if not query_results.empty:
            query_smiles = retrieve_massql_smiles(query_results, massql_spectra)
            query_fps = calculate_maccs_parallel(query_smiles)
            if motif.get("short_annotation"): # not checked yet
                annotation = motif.get("short_annotation") # new
                motif_fp = calc_fingerprints([[annotation]])[0] # new
                #print(calc_fingerprints([[motif.get("short_annotation")]][0]))
                #motif_fp = calc_fingerprints([[motif.get("short_annotation")]][0]) #new
            else: # new
                motif_fp = calc_fingerprints([motif.get("auto_annotation")])[0]
    
            sos = 0
            for query_fp in query_fps:
                sos += calculate_sos(motif_fp, query_fp)
            average_sos = sos / len(query_fps)
            matching_scores.append(average_sos)
        else:
            print("no result found!")
            print(motif.metadata)
            matching_scores.append(np.nan)
            continue

    grouped_ms2_motifDB["matching_score"] = matching_scores
    #colored_ms2_motifDB = grouped_ms2_motifDB.style.map(color_gradient, subset=["matching_score"])
    #return colored_ms2_motifDB
    return grouped_ms2_motifDB, ms1_spec, ms2_spec, queries



if __name__ == "__main__":
    calc_uniqueness_score()

    
