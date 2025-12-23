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
        result_df = temp_df[['region', 'data_type', 'value']]

        return jsonify({"status": "success", "data": result_df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)