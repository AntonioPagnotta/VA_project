import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATASET_FILE = 'combined_indicators_with_healthcare.csv'
CLUSTERING_FILE = 'clustering_results.json'
main_df = None
clustering_data = None
YEAR_LIMIT = 2051


def load_data():
    global main_df
    try:
        if not os.path.exists(DATASET_FILE):
            print(f"Error: {DATASET_FILE} not found.")
            return

        main_df = pd.read_csv(DATASET_FILE)
        main_df.columns = [c.strip().lower() for c in main_df.columns]

        if 'prev' in main_df.columns and 'pred' not in main_df.columns:
            main_df.rename(columns={'prev': 'pred'}, inplace=True)

        main_df['region'] = main_df['region'].astype(str).str.strip()
        main_df['data_type'] = main_df['data_type'].astype(str).str.strip()
        if 'indicator' in main_df.columns:
            main_df['indicator'] = main_df['indicator'].astype(str).str.strip()

        if 'year' in main_df.columns:
            main_df = main_df[main_df['year'] < YEAR_LIMIT]

        print("Dataset loaded successfully.")

    except Exception as e:
        print(f"Critical Error: {e}")


def load_clustering_data():
    global clustering_data
    try:
        if not os.path.exists(CLUSTERING_FILE):
            print(f"Warning: {CLUSTERING_FILE} not found. Run clustering_precompute.py first.")
            clustering_data = {}
            return
        
        with open(CLUSTERING_FILE, 'r') as f:
            clustering_data = json.load(f)
        
        print(f"Clustering data loaded successfully. {len(clustering_data)} configurations available.")
    
    except Exception as e:
        print(f"Error loading clustering data: {e}")
        clustering_data = {}


load_data()
load_clustering_data()


def filter_by_pred(df, pred_val):
    if pred_val is None or pred_val == -1:
        return df
    if 'pred' in df.columns:
        return df[df['pred'] == pred_val]
    return df


@app.route('/indicators', methods=['GET'])
def get_indicators():
    global main_df
    if main_df is None: return jsonify({"status": "error"}), 500

    try:
        req_pred = request.args.get('pred', None)
        pred_val = int(req_pred) if req_pred is not None else -1

        temp_df = filter_by_pred(main_df, pred_val)

        indicators = []
        if not temp_df.empty and 'indicator' in temp_df.columns and 'data_type' in temp_df.columns:
            ind_df = temp_df[['data_type', 'indicator']].drop_duplicates().sort_values('indicator')
            indicators = ind_df.rename(columns={'data_type': 'code', 'indicator': 'label'}).to_dict(orient='records')

        min_year = int(temp_df['year'].min()) if not temp_df.empty else 2022
        max_year = int(temp_df['year'].max()) if not temp_df.empty else 2022

        return jsonify({
            "status": "success",
            "indicators": indicators,
            "years": {"min": min_year, "max": max_year}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/query', methods=['GET'])
def query_data():
    global main_df
    if main_df is None: return jsonify({"status": "error"}), 500

    try:
        req_indicator = request.args.get('indicator', '')
        req_pred = request.args.get('pred', None)
        pred_val = int(req_pred) if req_pred is not None else -1

        mask = (main_df['data_type'] == req_indicator)
        if 'gender' in main_df.columns: mask &= (main_df['gender'] == 'total')

        temp_df = filter_by_pred(main_df[mask], pred_val)
        result_df = temp_df[['region', 'year', 'value']]

        return jsonify({"status": "success", "data": result_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/yearly_data', methods=['GET'])
def get_yearly_data():
    global main_df
    if main_df is None: return jsonify({"status": "error"}), 500

    try:
        req_year = int(request.args.get('year', 2022))
        req_pred = request.args.get('pred', None)
        pred_val = int(req_pred) if req_pred is not None else -1

        mask = (main_df['year'] == req_year)
        temp_df = filter_by_pred(main_df[mask], pred_val)
        
        # Include 'indicator' column if it exists
        if 'indicator' in temp_df.columns:
            result_df = temp_df[['region', 'data_type', 'indicator', 'value']]
        else:
            result_df = temp_df[['region', 'data_type', 'value']]

        return jsonify({"status": "success", "data": result_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/clustering', methods=['GET'])
def get_clustering():
    global clustering_data
    if clustering_data is None or len(clustering_data) == 0:
        return jsonify({
            "status": "error", 
            "message": "Clustering data not available. Run clustering_precompute.py first."
        }), 500

    try:
        year = request.args.get('year', 'overall')
        method = request.args.get('method', 'pca')  # pca or tsne
        clustering = request.args.get('clustering', 'kmeans')  # kmeans or spectral
        
        # Build key
        key = f"{year}_{method}_{clustering}"
        
        if key not in clustering_data:
            return jsonify({
                "status": "error",
                "message": f"Configuration not found: {key}"
            }), 404
        
        result = clustering_data[key]
        
        return jsonify({
            "status": "success",
            **result
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/clustering_timeseries', methods=['GET'])
def get_clustering_timeseries():
    global clustering_data
    if clustering_data is None or len(clustering_data) == 0:
        return jsonify({
            "status": "error",
            "message": "Clustering data not available. Run clustering_precompute.py first."
        }), 500

    try:
        method = request.args.get('method', 'pca')
        clustering = request.args.get('clustering', 'kmeans')
        
        # Extract all year-based configurations
        timeseries = []
        for key, data in clustering_data.items():
            if key.startswith('overall'):
                continue
            
            parts = key.split('_')
            if len(parts) >= 3:
                year_str = parts[0]
                key_method = parts[1]
                key_clustering = parts[2]
                
                if key_method == method and key_clustering == clustering:
                    try:
                        year_int = int(year_str)
                        timeseries.append({
                            'year': year_int,
                            **data
                        })
                    except ValueError:
                        continue
        
        # Sort by year
        timeseries.sort(key=lambda x: x['year'])
        
        return jsonify({
            "status": "success",
            "timeseries": timeseries,
            "count": len(timeseries)
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/clustering_manual', methods=['GET'])
def get_clustering_manual():
    """
    Compute clustering on-the-fly with specified K
    """
    global main_df
    if main_df is None:
        return jsonify({"status": "error", "message": "Dataset not loaded"}), 500
    
    try:
        year = request.args.get('year', 'overall')
        method = request.args.get('method', 'pca')
        clustering_type = request.args.get('clustering', 'kmeans')
        k = int(request.args.get('k', 3))
        
        # Import required libraries
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Prepare data
        if year == 'overall':
            df_year = main_df.groupby(['region', 'data_type'])['value'].median().reset_index()
        else:
            year_int = int(year)
            df_year = main_df[main_df['year'] == year_int].groupby(['region', 'data_type'])['value'].mean().reset_index()
        
        # Pivot to get regions x indicators
        pivot = df_year.pivot(index='region', columns='data_type', values='value')
        
        # Filter regions with at least 80% of indicators present
        threshold = len(pivot.columns) * 0.8
        pivot_clean = pivot.dropna(thresh=threshold)
        pivot_clean = pivot_clean.fillna(pivot_clean.median())
        
        regions = pivot_clean.index.tolist()
        features = pivot_clean.values
        feature_names = pivot_clean.columns.tolist()
        
        # Dimensionality reduction
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            transformed = reducer.fit_transform(scaled_data)
            explained_var = reducer.explained_variance_ratio_.tolist()
        else:  # tsne
            reducer = TSNE(n_components=2, perplexity=min(30, len(features) - 1), 
                          max_iter=1000, random_state=42)
            transformed = reducer.fit_transform(scaled_data)
            explained_var = None
        
        # Clustering with specified K
        if clustering_type == 'kmeans':
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(transformed)
            centroids = clusterer.cluster_centers_.tolist()
        else:  # spectral
            clusterer = SpectralClustering(n_clusters=k, random_state=42, affinity='rbf')
            labels = clusterer.fit_predict(transformed)
            # Compute centroids manually
            centroids = []
            for i in range(k):
                cluster_points = transformed[labels == i]
                if len(cluster_points) > 0:
                    centroids.append(cluster_points.mean(axis=0).tolist())
                else:
                    centroids.append([0, 0])
        
        return jsonify({
            "status": "success",
            "regions": regions,
            "points": transformed.tolist(),
            "clusters": [int(c) for c in labels],
            "k_optimal": k,
            "centroids": centroids,
            "explained_variance": explained_var,
            "year": year if year == 'overall' else int(year),
            "feature_names": feature_names
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)