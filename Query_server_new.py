import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATASET_FILE = 'combined_indicators_with_healthcare.csv'
main_df = None
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


load_data()


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
        columns = ['region', 'data_type', 'value']
        if 'indicator' in temp_df.columns:
            columns.append('indicator')
        
        result_df = temp_df[columns]

        return jsonify({"status": "success", "data": result_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ==========================================
# CLUSTERING ENDPOINTS
# ==========================================

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

CLUSTERING_FILE = 'clustering_results.json'
clustering_data = {}

def load_clustering_results():
    global clustering_data
    try:
        if os.path.exists(CLUSTERING_FILE):
            with open(CLUSTERING_FILE, 'r') as f:
                clustering_data = json.load(f)
            print("Clustering results loaded successfully.")
        else:
            print(f"Warning: {CLUSTERING_FILE} not found.")
    except Exception as e:
        print(f"Error loading clustering results: {e}")

# Load clustering data on startup
load_clustering_results()


def perform_clustering_with_k(year, method='pca', clustering='kmeans', k=None):
    """
    Perform clustering on-the-fly with a specific K value.
    Returns a dict with regions, points, clusters, k_optimal (which will be k_used in this case)
    """
    global main_df
    
    if main_df is None:
        raise Exception("Data not loaded")
    
    # Get data for the specified year
    year_val = int(year) if year != 'overall' else None
    
    if year_val:
        year_data = main_df[main_df['year'] == year_val]
    else:
        # For 'overall', use 2022 or latest available year
        year_data = main_df[main_df['year'] == 2022]
    
    if year_data.empty:
        raise Exception(f"No data available for year {year}")
    
    # Pivot data: regions x indicators
    pivot_df = year_data.pivot_table(
        index='region',
        columns='data_type',
        values='value',
        aggfunc='mean'
    )
    
    print(f"📊 Pivot shape before cleaning: {pivot_df.shape} (regions x indicators)")
    print(f"📍 Regions before cleaning: {list(pivot_df.index)}")
    
    # Fill NaNs with column mean (don't drop regions)
    pivot_df = pivot_df.fillna(pivot_df.mean())
    
    # If still have NaNs (column with all NaNs), fill with 0
    pivot_df = pivot_df.fillna(0)
    
    print(f"📊 Pivot shape after cleaning: {pivot_df.shape}")
    print(f"📍 Regions after cleaning: {list(pivot_df.index)}")
    
    regions = pivot_df.index.tolist()
    X = pivot_df.values
    
    # Apply PCA for dimensionality reduction
    if method == 'pca':
        pca = PCA(n_components=2)
        points = pca.fit_transform(X)
    else:
        # If other methods are needed, add them here
        points = X[:, :2]  # Just take first 2 dimensions
    
    # Perform clustering with specified K
    if k is None:
        k = 4  # Default K
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(points)
    
    return {
        'regions': regions,
        'points': points.tolist(),
        'clusters': clusters.tolist(),
        'k_optimal': k,  # In manual mode, this is the requested K
        'k_used': k,  # Actual K used
        'year': year if year != 'overall' else 'overall'
    }


@app.route('/clustering', methods=['GET'])
def get_clustering_auto():
    """Auto clustering - uses precomputed results with optimal K"""
    global clustering_data
    try:
        method = request.args.get('method', 'pca')
        clustering = request.args.get('clustering', 'kmeans')
        year = request.args.get('year', 'overall')
        
        # Construct key
        key = f"{year}_{method}_{clustering}"
        
        if key in clustering_data:
            result = clustering_data[key].copy()
            if 'year' not in result:
                result['year'] = year
            result['status'] = 'success'
            return jsonify(result)
        else:
            return jsonify({"status": "error", "message": f"Clustering data not found for {key}"}), 404
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/clustering_manual', methods=['GET'])
@app.route('/clustering_manual', methods=['GET'])
def get_clustering_manual():
    """Manual clustering - computes clustering on-the-fly with specified K"""
    try:
        method = request.args.get('method', 'pca')
        clustering_type = request.args.get('clustering', 'kmeans')
        year = request.args.get('year', 'overall')
        k_param = request.args.get('k', None)
        
        if k_param is None:
            return jsonify({"status": "error", "message": "Parameter 'k' is required for manual clustering"}), 400
        
        try:
            k = int(k_param)
            if k < 2 or k > 10:
                return jsonify({"status": "error", "message": "K must be between 2 and 10"}), 400
        except ValueError:
            return jsonify({"status": "error", "message": "K must be a valid integer"}), 400
        
        print(f"🎯 Manual clustering requested: year={year}, method={method}, clustering={clustering_type}, K={k}")
        
        # Perform clustering with specified K
        result = perform_clustering_with_k(year, method, clustering_type, k)
        result['status'] = 'success'
        
        print(f"✅ Manual clustering completed: {len(result['regions'])} regions, {len(set(result['clusters']))} unique clusters")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error in manual clustering: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/clustering_timeseries', methods=['GET'])
def get_clustering_timeseries():
    global clustering_data
    try:
        method = request.args.get('method', 'pca')
        clustering = request.args.get('clustering', 'kmeans')
        # Collect all precomputed clustering entries that match the requested method and clustering
        timeseries = []
        for key, value in clustering_data.items():
            # Expect keys like '2002_pca_kmeans' or 'overall_pca_kmeans'
            try:
                parts = key.split('_')
                year_part = parts[0]
                method_part = parts[1] if len(parts) > 1 else ''
                clustering_part = parts[2] if len(parts) > 2 else ''
            except Exception:
                continue

            if method_part == method and clustering_part == clustering:
                # Only numeric years (skip 'overall')
                if year_part.isdigit():
                    entry = value.copy()
                    entry['year'] = int(year_part)
                    timeseries.append(entry)

        # Sort by year ascending
        timeseries.sort(key=lambda x: x.get('year', 0))

        if not timeseries:
            return jsonify({"status": "error", "message": "No timeseries data found"}), 404

        return jsonify({
            "status": "success",
            "timeseries": timeseries
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)