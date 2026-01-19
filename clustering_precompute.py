import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import json
import warnings
warnings.filterwarnings('ignore')

DATASET_FILE = 'combined_indicators_with_healthcare.csv'
OUTPUT_FILE = 'clustering_results.json'
YEAR_LIMIT = 2051
K_MAX = 8
K_MIN = 2

def load_and_pivot_data(year=None):
    """
    Load CSV and pivot to create region x indicators matrix
    Returns: regions list, feature matrix, feature names, year
    """
    df = pd.read_csv(DATASET_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    df['region'] = df['region'].astype(str).str.strip()
    df['data_type'] = df['data_type'].astype(str).str.strip()
    
    # Filter by year limit
    df = df[df['year'] < YEAR_LIMIT]
    
    if year is not None:
        # Single year - handle overlap by taking mean
        df_year = df[df['year'] == year].groupby(['region', 'data_type'])['value'].mean().reset_index()
    else:
        # Overall - use median across all years
        df_year = df.groupby(['region', 'data_type'])['value'].median().reset_index()
    
    # Pivot to get regions x indicators
    pivot = df_year.pivot(index='region', columns='data_type', values='value')
    
    # Filter regions with at least 80% of indicators present
    threshold = len(pivot.columns) * 0.8
    pivot_clean = pivot.dropna(thresh=threshold)
    
    # Fill remaining NaN with column median
    pivot_clean = pivot_clean.fillna(pivot_clean.median())
    
    regions = pivot_clean.index.tolist()
    feature_matrix = pivot_clean.values
    feature_names = pivot_clean.columns.tolist()
    
    return regions, feature_matrix, feature_names, year

def find_optimal_k_elbow(data, k_min=K_MIN, k_max=K_MAX):
    """
    Use Elbow Method with KneeLocator (kneedle algorithm) to find optimal k
    """
    inertias = []
    k_range = range(k_min, k_max + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Use kneedle algorithm to find elbow
    try:
        kn = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow is not None else 3
    except:
        optimal_k = 3  # Default fallback
    
    return optimal_k, list(inertias)

def find_optimal_k_spectral(data, k_min=K_MIN, k_max=K_MAX):
    """
    Use eigenvalue gap for Spectral Clustering
    """
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Compute affinity matrix
    affinity = rbf_kernel(data, gamma=1.0)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(affinity)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    # Find largest gap in first k_max eigenvalues
    gaps = []
    for i in range(k_min - 1, min(k_max, len(eigenvalues) - 1)):
        gap = eigenvalues[i] - eigenvalues[i + 1]
        gaps.append(gap)
    
    if gaps:
        optimal_k = k_min + np.argmax(gaps)
    else:
        optimal_k = 3
    
    return optimal_k, eigenvalues[:k_max+1].tolist()

def perform_dimensionality_reduction(data, method='pca'):
    """
    Apply PCA or t-SNE
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        transformed = reducer.fit_transform(scaled_data)
        explained_var = reducer.explained_variance_ratio_.tolist()
    else:  # tsne
        reducer = TSNE(n_components=2, perplexity=min(30, len(data) - 1), 
                      max_iter=1000, random_state=42)
        transformed = reducer.fit_transform(scaled_data)
        explained_var = None
    
    return transformed, explained_var

def perform_clustering(data, method='kmeans', k=3):
    """
    Apply K-means or Spectral clustering
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(data)
        centroids = clusterer.cluster_centers_.tolist()
    else:  # spectral
        clusterer = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
        labels = clusterer.fit_predict(data)
        # Compute centroids manually for spectral
        centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroids.append(cluster_points.mean(axis=0).tolist())
            else:
                centroids.append([0, 0])
    
    return labels.tolist(), centroids

def match_cluster_labels(current_centroids, previous_centroids):
    """
    Match cluster labels to previous timestep based on centroid distances
    Returns: mapping dict {old_label: new_label}
    """
    if previous_centroids is None or len(previous_centroids) == 0:
        return {i: i for i in range(len(current_centroids))}
    
    current = np.array(current_centroids)
    previous = np.array(previous_centroids)
    
    # Handle different number of clusters
    if len(current) != len(previous):
        # Just return identity mapping if cluster count changed
        return {i: i for i in range(len(current_centroids))}
    
    # Compute distance matrix
    from scipy.spatial.distance import cdist
    distances = cdist(current, previous)
    
    # Hungarian algorithm for optimal matching
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    
    mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    return mapping

def process_all_configurations():
    """
    Main processing function - compute all configurations
    """
    results = {}
    
    # Get all unique years
    df = pd.read_csv(DATASET_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[df['year'] < YEAR_LIMIT]
    years = sorted(df['year'].unique())
    
    print(f"Processing {len(years)} years from {years[0]} to {years[-1]}")
    print(f"Year limit: {YEAR_LIMIT}")
    
    methods = ['pca', 'tsne']
    clustering_types = ['kmeans', 'spectral']
    
    # Track centroids for temporal consistency
    centroid_tracker = {
        f"{method}_{clust}": None 
        for method in methods 
        for clust in clustering_types
    }
    
    # Process overall (median) first
    print("\nProcessing overall (median)...")
    regions, features, feature_names, _ = load_and_pivot_data(year=None)
    print(f"  Regions: {len(regions)}, Features: {len(feature_names)}")
    
    for method in methods:
        print(f"  Method: {method.upper()}")
        transformed, explained_var = perform_dimensionality_reduction(features, method)
        
        for clust_type in clustering_types:
            print(f"    Clustering: {clust_type}")
            
            # Find optimal k
            if clust_type == 'kmeans':
                optimal_k, inertias = find_optimal_k_elbow(transformed)
            else:
                optimal_k, eigenvals = find_optimal_k_spectral(features)
            
            print(f"      Optimal K: {optimal_k}")
            
            # Perform clustering
            labels, centroids = perform_clustering(transformed, clust_type, optimal_k)
            
            # Store results (convert numpy types to Python native types)
            key = f"overall_{method}_{clust_type}"
            results[key] = {
                'regions': regions,
                'points': transformed.tolist(),
                'clusters': [int(c) for c in labels],
                'k_optimal': int(optimal_k),
                'centroids': centroids,
                'explained_variance': explained_var,
                'year': 'overall',
                'feature_names': feature_names,
                'inertias': inertias if clust_type == 'kmeans' else None,
                'eigenvalues': eigenvals if clust_type == 'spectral' else None
            }
    
    # Process each year
    for year in years:
        print(f"\nProcessing year {year}...")
        try:
            regions, features, feature_names, _ = load_and_pivot_data(year=year)
            
            if len(regions) < K_MIN:
                print(f"  Skipping - insufficient data ({len(regions)} regions)")
                continue
            
            for method in methods:
                transformed, explained_var = perform_dimensionality_reduction(features, method)
                
                for clust_type in clustering_types:
                    tracker_key = f"{method}_{clust_type}"
                    
                    # Find optimal k
                    if clust_type == 'kmeans':
                        optimal_k, inertias = find_optimal_k_elbow(transformed)
                    else:
                        optimal_k, eigenvals = find_optimal_k_spectral(features)
                    
                    # Perform clustering
                    labels, centroids = perform_clustering(transformed, clust_type, optimal_k)
                    
                    # Match labels to previous year for consistency
                    if centroid_tracker[tracker_key] is not None:
                        mapping = match_cluster_labels(centroids, centroid_tracker[tracker_key])
                        # Remap labels
                        labels = [mapping.get(label, label) for label in labels]
                        # Remap centroids
                        new_centroids = [None] * len(centroids)
                        for old_idx, new_idx in mapping.items():
                            if new_idx < len(new_centroids):
                                new_centroids[new_idx] = centroids[old_idx]
                        centroids = [c if c is not None else [0, 0] for c in new_centroids]
                    
                    # Update tracker
                    centroid_tracker[tracker_key] = centroids
                    
                    # Store results (convert numpy types to Python native types)
                    key = f"{year}_{method}_{clust_type}"
                    results[key] = {
                        'regions': regions,
                        'points': transformed.tolist(),
                        'clusters': [int(c) for c in labels],
                        'k_optimal': int(optimal_k),
                        'centroids': centroids,
                        'explained_variance': explained_var,
                        'year': int(year),
                        'feature_names': feature_names,
                        'inertias': inertias if clust_type == 'kmeans' else None,
                        'eigenvalues': eigenvals if clust_type == 'spectral' else None
                    }
        
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue
    
    return results

if __name__ == '__main__':
    print("=" * 60)
    print("CLUSTERING PRECOMPUTATION")
    print("=" * 60)
    
    if not os.path.exists(DATASET_FILE):
        print(f"ERROR: {DATASET_FILE} not found!")
        exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  K range: {K_MIN} to {K_MAX}")
    print(f"  Year limit: < {YEAR_LIMIT}")
    
    print("\nStarting processing...")
    results = process_all_configurations()
    
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Processed {len(results)} configurations")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")