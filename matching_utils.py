"""
Matching logic, kept in its own module so that worker processes can import it.
(Multiprocessing needs functions that live in an importable module, not inside
the Streamlit script itself.)

Only plain numpy arrays are passed around here, which keeps the data that has
to be sent to the worker processes small and simple.
"""
import numpy as np


def all_within_tolerance(query_vals, target_vals, tolerance):
    """True if EVERY value in query_vals has a value in target_vals within `tolerance`."""
    if len(query_vals) == 0:
        return True
    if len(target_vals) == 0:
        return False
    # Matrix of all pairwise differences: rows = query values, columns = target values
    diffs = np.abs(query_vals[:, None] - target_vals[None, :])
    # For each query value: is there at least one target within tolerance? Then: are all found?
    return bool((diffs <= tolerance).any(axis=1).all())


def find_matches(chunk, alerts, frag_tol, loss_tol):
    """
    chunk  : list of (spectrum_index, fragment_mzs, loss_mzs) for the sample spectra
    alerts : list of (fragment_mzs, loss_mzs) for the spectral alerts
    Returns a list of (spectrum_index, alert_index) for every alert whose fragments
    AND losses are all contained in the sample spectrum.
    """
    hits = []
    for spec_idx, frags, losses in chunk:
        for alert_idx, (a_frags, a_losses) in enumerate(alerts):
            if (all_within_tolerance(a_frags, frags, frag_tol)
                    and all_within_tolerance(a_losses, losses, loss_tol)):
                hits.append((spec_idx, alert_idx))
    return hits
