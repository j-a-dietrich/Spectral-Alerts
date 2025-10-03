from MS2LDA.Add_On.Spec2Vec.annotation_refined import calc_embeddings

import numpy as np
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
#from sklearn.metrics.pairwise import cosine_distance



# ---------------- Clustering Function ----------------
def cluster_motifs(motif_candidates, s2v_similarity, distance_threshold=0.95, min_motifs_per_cluster=5):
    """
    Clusters motifs using agglomerative clustering on spectral embeddings from Spec2Vec.
    
    PARAMETERS:
        motif_candidates (dict): Dictionary of motif objects (also used as unoptimized motifs).
        s2v_similarity: Spec2Vec model.
        distance_threshold (float): Higher value = stricter clustering.
        min_motifs_per_cluster (int): Minimum motifs per cluster.

    RETURNS:
        filtered_clusters (dict): {cluster_id: list of motif_ids}
    """
    spectral_embeddings = calc_embeddings(s2v_similarity, motif_candidates)
    #distance_matrix = cosine_distance(spectral_embeddings)
    agc = AgglomerativeClustering(
        distance_threshold=distance_threshold, n_clusters=None, metric="cosine", linkage="complete"
    )
    cluster_ids = agc.fit_predict(spectral_embeddings)

    motifs_per_cluster = Counter(cluster_ids)
    filtered_clusters = {cluster_id: [] for cluster_id, n_motifs in motifs_per_cluster.items() if n_motifs >= min_motifs_per_cluster}

    for motif_id, cluster_id in enumerate(cluster_ids):
        if cluster_id in filtered_clusters:
            filtered_clusters[cluster_id].append(motif_id)

    return filtered_clusters

# ---------------- Ranking Functions ----------------
def rank_by_spec_optimization(motifs, motif_candidates, optimized_motif_candidates):
    """Ranks motifs by spectral optimization score (higher is better)."""
    return {
        motif_id: rank for rank, motif_id in enumerate(sorted(
            motifs,
            key=lambda motif_id: (
                np.sum(optimized_motif_candidates[motif_id].peaks.intensities) /
                np.sum(motif_candidates[motif_id].peaks.intensities)  # Updated to use motif_candidates
            ),
            reverse=True
        ), start=1)
    }

def rank_by_n_annotations(motifs, optimized_motif_candidates):
    """Ranks motifs by number of annotations (higher is better)."""
    return {
        motif_id: rank for rank, motif_id in enumerate(sorted(
            motifs,
            key=lambda motif_id: len(optimized_motif_candidates[motif_id].get("auto_annotation", [])),
            reverse=True
        ), start=1)
    }

def rank_by_annotation_coherence(motifs, fps_motif_candidates):
    """Ranks motifs by annotation coherence score (higher is better)."""
    return {
        motif_id: rank for rank, motif_id in enumerate(sorted(
            motifs,
            key=lambda motif_id: np.sum(fps_motif_candidates[motif_id]),
            reverse=True
        ), start=1)
    }

def rank_by_n_fragments(motifs, optimized_motif_candidates):
    """Ranks motifs by annotation coherence score (higher is better)."""
    return {
        motif_id: rank for rank, motif_id in enumerate(sorted(
            motifs,
            key=lambda motif_id: len(optimized_motif_candidates[motif_id].peaks.mz),
            reverse=True
        ), start=1)
    }


# ----------------- Filtering Functions ---------------------------
def filter_by_spec_optimization(filtered_clusters, motif_candidates, optimized_motif_candidates, min_explained=0.5):
    """
    Filters motifs based on spectral optimization score without ranking.

    PARAMETERS:
    - filtered_clusters (dict): {cluster_id: list of motif_ids}
    - motif_candidates (dict): Dictionary containing unoptimized motif data.
    - optimized_motif_candidates (dict): Dictionary containing optimized motif data.
    - min_explained (float): Minimum required spectral optimization score.

    RETURNS:
    - filtered_clusters (dict): {cluster_id: list of filtered motif_ids}
    """

    for cluster_id in list(filtered_clusters.keys()):
        filtered_clusters[cluster_id] = [
            motif_id for motif_id in filtered_clusters[cluster_id]
            if (
                np.sum(optimized_motif_candidates[motif_id].peaks.intensities) /
                np.sum(motif_candidates[motif_id].peaks.intensities)
            ) >= min_explained
        ]

    return filtered_clusters



def filter_by_n_annotations(filtered_clusters, optimized_motif_candidates, min_annotations=2):
    """
    Filters motifs based on the number of annotations without ranking.

    PARAMETERS:
    - filtered_clusters (dict): {cluster_id: list of motif_ids}
    - optimized_motif_candidates (dict): Dictionary containing motif data with annotations.
    - min_annotations (int): Minimum required number of annotations.

    RETURNS:
    - filtered_clusters (dict): {cluster_id: list of filtered motif_ids}
    """

    for cluster_id in list(filtered_clusters.keys()):
        filtered_clusters[cluster_id] = [
            motif_id for motif_id in filtered_clusters[cluster_id]
            if len(optimized_motif_candidates[motif_id].get("auto_annotation", [])) >= min_annotations
        ]

    return filtered_clusters


def filter_by_n_fragments(filtered_clusters, optimized_motif_candidates, min_fragments=2):
    """
    Filters motifs based on the number of fragments (peaks + losses).

    PARAMETERS:
    - filtered_clusters (dict): {cluster_id: list of motif_ids}
    - optimized_motif_candidates (dict): Dictionary containing motif data with annotations.
    - min_fragments (int): Minimum required number of fragments.

    RETURNS:
    - filtered_clusters (dict): {cluster_id: list of filtered motif_ids}
    """

    for cluster_id in list(filtered_clusters.keys()):
        filtered_clusters[cluster_id] = [
            motif_id for motif_id in filtered_clusters[cluster_id]
            if (
                len(getattr(optimized_motif_candidates[motif_id].peaks, 'mz', [])) +
                len(getattr(optimized_motif_candidates[motif_id].losses, 'mz', []))
            ) >= min_fragments
        ]

    return filtered_clusters



def filter_by_annotation_coherence(filtered_clusters, fps_motif_candidates, min_coherence=1):
    """
    Filters motifs based on annotation coherence score without ranking.

    PARAMETERS:
    - filtered_clusters (dict): {cluster_id: list of motif_ids}
    - fps_motif_candidates (dict): Dictionary containing fingerprint data for annotation coherence.
    - min_coherence (float): Minimum required annotation coherence score.

    RETURNS:
    - filtered_clusters (dict): {cluster_id: list of filtered motif_ids}
    """

    for cluster_id in list(filtered_clusters.keys()):
        filtered_clusters[cluster_id] = [
            motif_id for motif_id in filtered_clusters[cluster_id]
            if np.sum(fps_motif_candidates[motif_id]) >= min_coherence
        ]

    return filtered_clusters



# ---------------- Final Ranking Computation ----------------
def compute_final_scores(motifs, rank_spec_opt, rank_n_ann, rank_ann_coh, rank_n_frag,
                         weight_spec_opt=0.333, weight_n_ann=0.333, weight_ann_coh=0.333, weight_n_frag=0.333):
    """
    Computes final weighted scores for motifs.

    RETURNS:
        final_scores (dict): {motif_id: final_score}
    """
    return {
        motif_id: (
            weight_spec_opt * rank_spec_opt.get(motif_id, 0) +
            weight_n_ann * rank_n_ann.get(motif_id, 0) +
            weight_ann_coh * rank_ann_coh.get(motif_id, 0) +
            weight_n_frag * rank_n_frag.get(motif_id, 0)
        )
        for motif_id in motifs
    }



def compute_final_ranking(filtered_clusters, motif_candidates, optimized_motif_candidates, fps_motif_candidates,
                          weight_spec_opt=0.333, weight_n_ann=0.333, weight_ann_coh=0.333, weight_n_frag=0.333):
    """
    Computes final motif ranking per cluster.

    RETURNS:
        final_ranked_clusters (dict): {cluster_id: sorted list of motif_ids}
    """
    final_ranked_clusters = {}

    for cluster_id, motifs in filtered_clusters.items():
        # Compute rankings
        rank_spec_opt = rank_by_spec_optimization(motifs, motif_candidates, optimized_motif_candidates)
        rank_n_ann = rank_by_n_annotations(motifs, optimized_motif_candidates)
        rank_n_frag = rank_by_n_fragments(motifs, optimized_motif_candidates)
        rank_ann_coh = rank_by_annotation_coherence(motifs, fps_motif_candidates)
        

        # Compute final scores
        final_scores = compute_final_scores(motifs, rank_spec_opt, rank_n_ann, rank_ann_coh, rank_n_frag, 
                                            weight_spec_opt, weight_n_ann, weight_ann_coh, weight_n_frag)

        # Sort motifs by final score (lower is better)
        final_ranked_clusters[cluster_id] = sorted(final_scores.keys(), key=lambda m: final_scores[m])

    return final_ranked_clusters

def remove_nan_motifs(motif_candidates, optimized_motif_candidates, fps_motif_candidates):
    """removes motifs without any losses and fragments"""
    assert len(motif_candidates) == len(optimized_motif_candidates)
    assert len(motif_candidates) == len(fps_motif_candidates)
    
    valid_optimized_motif_candidates = []
    valid_motif_candidates = []
    valid_fps_motif_candidates = []

    for i in range(len(motif_candidates)):

        if optimized_motif_candidates[i].peaks or optimized_motif_candidates[i].losses:

            valid_optimized_motif_candidates.append(optimized_motif_candidates[i])
            valid_motif_candidates.append(motif_candidates[i])
            valid_fps_motif_candidates.append(fps_motif_candidates[i])
    
    return valid_motif_candidates, valid_optimized_motif_candidates, valid_fps_motif_candidates


# ---------------- Full Workflow ----------------
def run_motif_clustering_and_ranking(motif_candidates, s2v_similarity, optimized_motif_candidates, fps_motif_candidates,
                                     distance_threshold=0.95, min_motifs_per_cluster=5,
                                     min_explained=0.5, min_annotations=2, min_coherence=1, min_fragments=2,
                                     weight_spec_opt=0.333, weight_n_ann=0.333, weight_ann_coh=0.333, weight_n_frag=0.333):
    """
    Full pipeline: Clusters motifs, applies filtering, and computes final rankings.
    """
    # Step 0: Filter Optimized Motifs without fragments and losses
    # motif_candidates, optimized_motif_candidates, fps_motif_candidates = remove_nan_motifs(motif_candidates, optimized_motif_candidates, fps_motif_candidates)
    # DONE in the notebook! Otherwise it would be important to use the same filtered motifs for the plotting otherise the other is wrong

    # Step 1: Cluster motifs
    filtered_clusters = cluster_motifs(optimized_motif_candidates, s2v_similarity, distance_threshold, min_motifs_per_cluster)

    # Step 2: Apply filtering
    filtered_clusters = filter_by_spec_optimization(filtered_clusters, motif_candidates, optimized_motif_candidates, min_explained)
    filtered_clusters = filter_by_n_annotations(filtered_clusters, optimized_motif_candidates, min_annotations)
    filtered_clusters = filter_by_annotation_coherence(filtered_clusters, fps_motif_candidates, min_coherence)
    filtered_clusters = filter_by_n_fragments(filtered_clusters, optimized_motif_candidates, min_fragments)

    # Step 3: Compute rankings
    final_ranked_clusters = compute_final_ranking(
        filtered_clusters, motif_candidates, optimized_motif_candidates, fps_motif_candidates,
        weight_spec_opt, weight_n_ann, weight_ann_coh, weight_n_frag
    )

    return final_ranked_clusters



if __name__ == "__main__":
    final_clusters = run_motif_clustering_and_ranking(
        motif_candidates,
        s2v_similarity,
        optimized_motif_candidates,
        fps_motif_candidates,
        distance_threshold=0.9,
        min_motifs_per_cluster=5,
        weight_spec_opt=0.4,
        weight_n_ann=0.3,
        weight_ann_coh=0.3,
        weight_n_frag=0.3
    )
