import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
# Import render_template
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from statsmodels.tsa.stattools import adfuller
import numpy as np

# --- Matplotlib Configuration ---
plt.switch_backend('Agg')

# --- Initialize Flask App ---
# Point 'static_folder' to None if you don't have one,
# and specify the 'templates' folder.
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- NEW: Route to Serve the HTML Webpage ---
@app.route('/')
def home():
    """
    This new route serves your index.html file from the 'templates' folder.
    """
    return render_template('index.html')

# --- Helper Function: Create and Encode Plot ---
def create_plot_base64(series: pd.Series, title: str, xlabel: str = "Time", ylabel: str = "Value") -> str:
    """Generates a line plot from a pandas Series and returns it as a base64 encoded string."""
    try:
        plt.figure(figsize=(10, 6))
        series.plot(grid=True)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()

        # Save plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode buffer to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Close the plot to free memory (VERY important in a server)
        plt.close()
        
        return image_base64
    except Exception as e:
        plt.close() # Ensure plot is closed even on error
        print(f"Error creating plot: {e}")
        return None

# --- Helper Function: Run Stationarity Test ---
def check_stationarity(timeseries: pd.Series):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.
    
    """
    # The adfuller() test is a common statistical test for stationarity.
    # We are most interested in the p-value.
    try:
        result = adfuller(timeseries.dropna()) # Drop NA values for the test
        
        p_value = result[1]
        is_stationary = p_value < 0.05  # Common threshold for significance

        # FIX: Convert all numpy types to standard python types for JSON
        return {
            "adf_statistic": float(result[0]),
            "p_value": float(p_value),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary": bool(is_stationary),
            "interpretation": f"p-value: {p_value:.4f}. Data is {'stationary' if is_stationary else 'non-stationary'}."
        }
    except Exception as e:
        # Can fail if data is all constant, etc.
        return {
            "is_stationary": False,
            "interpretation": f"Test failed: {e}",
            "error": str(e)
        }

# --- API Endpoint 1: Inspect CSV ---
@app.route("/inspect-csv", methods=['POST'])
def inspect_csv():
    """
    Upload a CSV file to get a list of its column headers.
    This is the first step a user should take.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read only the first 5 rows to get headers and sample data quickly
        df = pd.read_csv(file, nrows=5)
        return jsonify({
            "filename": file.filename,
            "columns": list(df.columns),
            "sample_data": df.head().to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": f"Error processing CSV file: {e}"}), 400

# --- API Endpoint 2: Analyze Time Series ---
@app.route("/analyze", methods=['POST'])
def analyze_time_series():
    """
    Perform the full time series analysis on the selected columns.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    date_column = request.form.get('date_column')
    value_column = request.form.get('value_column')

    if not date_column or not value_column:
        return jsonify({"error": "Missing date_column or value_column"}), 400

    try:
        # Read the full CSV file
        # We need to seek back to the beginning of the file stream
        file.seek(0)
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {e}"}), 400

    # --- 1. Validate Columns ---
    if date_column not in df.columns or value_column not in df.columns:
        return jsonify({"error": f"Error: One or both columns ('{date_column}', '{value_column}') not found in CSV."}), 404

    # --- 2. Process and Clean Data ---
    try:
        # Convert value column to numeric, forcing errors to NaN
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        
        # Convert date column to datetime, forcing errors to NaT
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        # Drop rows where our key columns have bad data
        original_rows = len(df)
        df.dropna(subset=[date_column, value_column], inplace=True)
        cleaned_rows = len(df)
        
        if cleaned_rows == 0:
            return jsonify({"error": "No valid data remaining after cleaning date and value columns."}), 400
        
        # Set the date column as the index
        df.set_index(date_column, inplace=True)
        
        # Sort by date (crucial for time series)
        df.sort_index(inplace=True)
        
        # Select our final time series data
        ts = df[value_column]

    except Exception as e:
        return jsonify({"error": f"Error processing columns: {e}"}), 400

    # --- 3. Perform Analysis ---
    
    # Calculate Mean and Variance for original data
    original_mean = float(ts.mean())
    original_variance = float(ts.var())
    
    # Draw graph for original data
    original_plot_b64 = create_plot_base64(ts, f"Original Time Series: {value_column}")
    
    # Check for stationarity
    stationarity_test = check_stationarity(ts)

    # --- 4. Handle Non-Stationary Data ---
    stationary_results = None
    if not stationarity_test.get("is_stationary", False):
        try:
            # Convert to stationary using first-order differencing
            # 
            ts_stationary = ts.diff().dropna()
            
            if ts_stationary.empty:
                 stationary_results = {"error": "Differencing resulted in empty series."}
            else:
                # Calculate stats for new stationary data
                # FIX: Convert numpy types to standard float
                stationary_mean = float(ts_stationary.mean())
                stationary_variance = float(ts_stationary.var())
                
                # Draw graph for new stationary data
                stationary_plot_b64 = create_plot_base64(
                    ts_stationary, 
                    f"Stationary Time Series (1st Order Differencing)"
                )
                
                # Re-run stationarity test to confirm
                new_stationarity_test = check_stationarity(ts_stationary)
                
                stationary_results = {
                    "method": "First-Order Differencing",
                    "mean": stationary_mean,
                    "variance": stationary_variance,
                    "plot_base64": stationary_plot_b64,
                    "stationarity_test": new_stationarity_test
                }
        except Exception as e:
            # If differencing fails for some reason
            stationary_results = {"error": f"Could not convert to stationary: {e}"}

    # --- 5. Assemble Final Response ---
    response_data = {
        "analysis_summary": {
            "filename": file.filename,
            "date_column": date_column,
            "value_column": value_column,
            "data_cleaning": {
                "original_rows": int(original_rows),
                "cleaned_rows": int(cleaned_rows),
                "rows_dropped": int(original_rows - cleaned_rows)
            }
        },
        "original_data": {
            "mean": original_mean,
            "variance": original_variance,
            "plot_base64": original_plot_b64,
            "stationarity_test": stationarity_test
        },
        "stationary_results": stationary_results
    }
    
    return jsonify(response_data)


# This block allows you to run the app directly with `python app.py`
if __name__ == "__main__":
    app.run(debug=True)

