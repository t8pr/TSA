import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
import numpy as np
import re # Import regular expressions for cleaning

# --- IMPORTS ---
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Matplotlib Configuration ---
plt.switch_backend('Agg')

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Route to Serve the HTML Webpage ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# ===--- PLOTTING HELPER FUNCTIONS ---===
# (All plotting functions remain the same)

def create_plot_base64(series: pd.Series, title: str, xlabel: str = "Time", ylabel: str = "Value") -> str:
    """Generates a line plot from a pandas Series and returns it as a base64 encoded string."""
    try:
        plt.figure(figsize=(10, 6))
        series.plot(grid=True)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        plt.close()
        print(f"Error creating plot: {e}")
        return None

def create_lag_plot_base64(series: pd.Series) -> str:
    """Generates a lag plot and returns it as a base64 encoded string."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        lag_plot(series, ax=ax)
        ax.set_title('Lag Plot (y vs y-1)', fontsize=16)
        ax.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return image_base64
    except Exception as e:
        plt.close(fig if 'fig' in locals() else None)
        print(f"Error creating lag plot: {e}")
        return None

def create_rolling_stats_plot(series: pd.Series) -> str:
    """Generates a plot with rolling mean and rolling std."""
    try:
        plt.figure(figsize=(12, 7))
        window = min(30, len(series) // 4)
        if window < 2: window = 2
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        series.plot(color='blue', label='Original Series', grid=True)
        rolling_mean.plot(color='red', label=f'Rolling Mean (w={window})')
        rolling_std.plot(color='black', label=f'Rolling Std (w={window})')
        plt.title('Rolling Mean & Standard Deviation', fontsize=16)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        plt.close()
        print(f"Error creating rolling stats plot: {e}")
        return None

def create_decomposition_plot(result: 'DecomposeResult') -> str:
    """Generates a 4-panel decomposition plot."""
    try:
        fig = plt.figure(figsize=(10, 12))
        ax_obs = fig.add_subplot(411)
        result.observed.plot(ax=ax_obs, grid=True)
        ax_obs.set_ylabel('Observed')
        ax_trend = fig.add_subplot(412)
        result.trend.plot(ax=ax_trend, grid=True)
        ax_trend.set_ylabel('Trend')
        ax_seasonal = fig.add_subplot(413)
        result.seasonal.plot(ax=ax_seasonal, grid=True)
        ax_seasonal.set_ylabel('Seasonal')
        ax_resid = fig.add_subplot(414)
        result.resid.plot(ax=ax_resid, grid=True, marker='.')
        ax_resid.set_ylabel('Residual')
        fig.suptitle('Seasonal Decomposition', fontsize=18, y=1.02)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return image_base64
    except Exception as e:
        plt.close(fig if 'fig' in locals() else None)
        print(f"Error creating decomposition plot: {e}")
        return None

def create_acf_pacf_plots(series_diff: pd.Series) -> dict:
    """Generates ACF and PACF plots for a differenced series."""
    plots = {}
    try:
        nobs = len(series_diff)
        nlags = min(30, (nobs // 2) - 1)
        if nlags < 1: nlags = 1

        # ACF Plot
        fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
        plot_acf(series_diff, lags=nlags, ax=ax_acf)
        ax_acf.set_title('Autocorrelation Function (ACF) of Differenced Series')
        plt.tight_layout()
        buf_acf = io.BytesIO()
        fig_acf.savefig(buf_acf, format='png')
        buf_acf.seek(0)
        plots['acf_plot_base64'] = base64.b64encode(buf_acf.read()).decode('utf-8')
        plt.close(fig_acf)

        # PACF Plot
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 5))
        plot_pacf(series_diff, lags=nlags, method='ywm', ax=ax_pacf)
        ax_pacf.set_title('Partial Autocorrelation (PACF) of Differenced Series')
        plt.tight_layout()
        buf_pacf = io.BytesIO()
        fig_pacf.savefig(buf_pacf, format='png')
        buf_pacf.seek(0)
        plots['pacf_plot_base64'] = base64.b64encode(buf_pacf.read()).decode('utf-8')
        plt.close(fig_pacf)
        
        return plots
    except Exception as e:
        plt.close(fig_acf if 'fig_acf' in locals() else None)
        plt.close(fig_pacf if 'fig_pacf' in locals() else None)
        print(f"Error creating ACF/PACF plots: {e}")
        return {'error': str(e)}

def create_residual_plots(residuals: pd.Series) -> dict:
    """Generates diagnostic plots for ARIMA residuals."""
    plots = {}
    try:
        # Residual Time Plot & Density
        fig_res, (ax_res, ax_kde) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
        residuals.plot(ax=ax_res, title='Residuals Over Time', grid=True)
        ax_res.set_ylabel('Residual')
        residuals.plot(kind='kde', ax=ax_kde, title='Residual Density', grid=True)
        ax_kde.set_xlabel('Residual Value')
        plt.tight_layout()
        buf_res = io.BytesIO()
        fig_res.savefig(buf_res, format='png')
        buf_res.seek(0)
        plots['residual_plot_base64'] = base64.b64encode(buf_res.read()).decode('utf-8')
        plt.close(fig_res)
        
        nobs_res = len(residuals)
        nlags_res = min(30, (nobs_res // 2) - 1)
        if nlags_res < 1: nlags_res = 1
        
        # Residual ACF Plot
        fig_acf, ax_acf = plt.subplots(figsize=(10, 5))
        plot_acf(residuals, lags=nlags_res, ax=ax_acf)
        ax_acf.set_title('ACF of Residuals')
        plt.tight_layout()
        buf_acf = io.BytesIO()
        fig_acf.savefig(buf_acf, format='png')
        buf_acf.seek(0)
        plots['residual_acf_plot_base64'] = base64.b64encode(buf_acf.read()).decode('utf-8')
        plt.close(fig_acf)

        return plots
    except Exception as e:
        plt.close(fig_res if 'fig_res' in locals() else None)
        plt.close(fig_acf if 'fig_acf' in locals() else None)
        print(f"Error creating residual plots: {e}")
        return {'error': str(e)}

def create_forecast_plot(train: pd.Series, test: pd.Series, forecast: pd.Series) -> str:
    """Generates a plot showing train, test, and forecast data."""
    try:
        plt.figure(figsize=(12, 7))
        # Check if index is numeric (for non-datetime)
        is_numeric_index = pd.api.types.is_numeric_dtype(train.index)
        
        if is_numeric_index:
            train.plot(label='Train', grid=True)
            test.plot(label='Test (Actual)', color='orange')
            forecast.plot(label='Forecast', color='green', linestyle='--')
        else:
            # Handle plotting for string indexes like "Jan", "Feb"
            plt.plot(range(len(train)), train.values, label='Train')
            plt.plot(range(len(train), len(train) + len(test)), test.values, label='Test (Actual)', color='orange')
            plt.plot(range(len(train), len(train) + len(test)), forecast.values, label='Forecast', color='green', linestyle='--')
            
            # Create cleaner labels for non-datetime index
            combined_index = train.index.to_list() + test.index.to_list()
            tick_positions = np.linspace(0, len(combined_index) - 1, num=10, dtype=int)
            plt.xticks(ticks=tick_positions, labels=[combined_index[i] for i in tick_positions], rotation=45)

        plt.title('ARIMA Train/Test Forecast', fontsize=16)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        plt.close()
        print(f"Error creating forecast plot: {e}")
        return None

# ===--- ANALYSIS HELPER FUNCTIONS ---===

def check_stationarity(timeseries: pd.Series):
    """Performs the Augmented Dickey-Fuller test to check for stationarity."""
    if timeseries.empty:
        return {"error": "Time series is empty, cannot check stationarity."}
    try:
        result = adfuller(timeseries.dropna())
        p_value = result[1]
        is_stationary = p_value < 0.05

        return {
            "adf_statistic": float(result[0]),
            "p_value": float(p_value),
            "critical_values": result[4],
            "is_stationary": bool(is_stationary),
            "interpretation": f"p-value: {p_value:.4f}. Data is {'stationary' if is_stationary else 'non-stationary'}."
        }
    except Exception as e:
        print(f"Error in ADF test: {e}")
        return {
            "adf_statistic": None,
            "p_value": None,
            "critical_values": None,
            "is_stationary": False,
            "interpretation": f"ADF test failed: {e}.",
            "error": str(e)
        }

def run_decomposition_analysis(ts_interpolated: pd.Series):
    """Performs seasonal decomposition."""
    try:
        freq = pd.infer_freq(ts_interpolated.index)
        period = 7
        if freq and freq.startswith('M'):
            period = 12
        if freq and freq.startswith('Q'):
            period = 4
        
        if len(ts_interpolated) < 2 * period:
            return {"error": f"Series is too short for decomposition (needs at least {2*period} data points).", "period": period}

        decomposition = seasonal_decompose(ts_interpolated, model='additive', period=period)
        plot_base64 = create_decomposition_plot(decomposition)
        
        if plot_base64 is None:
            raise Exception("Plotting failed.")

        return {
            "plot_base64": plot_base64,
            "period": period,
            "method": "additive"
        }
    except Exception as e:
        print(f"Error in decomposition: {e}")
        return {"error": str(e), "period": None}

def run_simple_arima_analysis(ts_data: pd.Series, freq: str, is_datetime_index: bool):
    """
    Runs a simple ARIMA(1,1,1) model, performs train/test
    split, residual analysis, and returns the results.
    """
    results = {}
    try:
        # --- 1. Log Transform (if possible) ---
        use_log = (ts_data > 0).all()
        if use_log:
            ts_log = np.log(ts_data)
            results['log_transformed_message'] = "Log transform applied (data all positive)."
        else:
            ts_log = ts_data
            results['log_transformed_message'] = "Log transform not applied (some values <= 0)."
        
        # --- 2. Create differenced series for ACF/PACF ---
        d = 1
        adf_test = check_stationarity(ts_log)
        if adf_test.get('is_stationary', False):
            d = 0
            
        ts_diff = ts_log.diff(periods=d).dropna()
        
        # NEW: Conditional message
        if is_datetime_index:
            results['preprocessing_message'] = f"Data interpolated to '{freq}' frequency. Using d={d} for differencing."
        else:
            results['preprocessing_message'] = f"Using simple non-datetime index. Using d={d} for differencing. Interpolation skipped."
        
        # --- 3. ACF/PACF Plots ---
        acf_pacf_plots = create_acf_pacf_plots(ts_diff)
        if 'error' in acf_pacf_plots:
            raise Exception(acf_pacf_plots['error'])
        results.update(acf_pacf_plots)
            
        # --- 4. Train/Test Split (80/20) ---
        split_point = int(len(ts_log) * 0.8)
        train_log = ts_log.iloc[:split_point]
        test_log = ts_log.iloc[split_point:]
        
        if len(train_log) < 20 or len(test_log) < 1:
             return {"error": f"Not enough data for ARIMA train/test split."}

        # --- 5. Fit ARIMA(1,1,1) model ---
        # We pass train_log.values to avoid index issues with non-datetime data
        model_order = (1, 1, 1)
        model = ARIMA(train_log.values, order=model_order)
        model_fit = model.fit()

        # --- 6. Get Residuals & Plots ---
        residuals = pd.Series(model_fit.resid) # Residuals won't have a time index
        if not residuals.empty:
            residual_plots = create_residual_plots(residuals)
            results.update(residual_plots)
        else:
            print("Warning: No residuals found.")

        # --- 7. Forecast (on test set) ---
        forecast_log_values = model_fit.forecast(steps=len(test_log))
        # Create a new series with the *test index*
        forecast_log = pd.Series(forecast_log_values, index=test_log.index)
        
        # --- 8. Inverse Transform (if log was used) ---
        if use_log:
            train_real = np.exp(train_log)
            test_real = np.exp(test_log)
            forecast_real = np.exp(forecast_log)
        else:
            train_real = train_log
            test_real = test_log
            forecast_real = forecast_log
            
        forecast_real.index = test_real.index

        # --- 9. Calculate Metrics ---
        results['rmse'] = float(sqrt(mean_squared_error(test_real, forecast_real)))
        results['mae'] = float(mean_absolute_error(test_real, forecast_real))

        # --- 10. Generate Forecast Plot ---
        results['forecast_plot_base64'] = create_forecast_plot(train_real, test_real, forecast_real)
        
        return results

    except Exception as e:
        print(f"Simple ARIMA analysis failed: {e}")
        return {"error": f"Simple ARIMA analysis failed: {e}"}

# --- FINAL SUMMARY FUNCTION ---
def generate_final_summary(stationarity_test, decomposition_results, arima_results):
    """
    Generates a final text summary based on all analysis results.
    """
    try:
        title = ""
        recommendation = ""
        next_steps = ""
        
        is_stationary = stationarity_test.get('is_stationary', False)
        # NEW: Check if decomposition was skipped
        is_decomposed = 'error' not in decomposition_results
        has_arima_residuals = 'error' not in arima_results and 'residual_acf_plot_base64' in arima_results

        if is_stationary:
            # Case 1: Stationary Data
            title = "Recommendation: Use ARMA Model"
            recommendation = "The ADF test suggests your data is already **stationary (d=0)**. A simpler ARMA(p,q) model is likely sufficient. The 'differencing' step (d=1) in our trial model may be unnecessary."
            next_steps = "Look at the **ACF and PACF plots** of the *original* data (not shown) to determine the 'p' and 'q' orders for an ARMA model."
        
        elif not is_stationary and is_decomposed:
            # Case 2: Non-Stationary with Seasonality (requires datetime index)
            title = "Recommendation: Use SARIMA Model"
            recommendation = (
                "The ADF test shows your data is **non-stationary (d>0)**. "
                "More importantly, the Seasonal Decomposition plot shows a clear **seasonal pattern** "
                f"(with a period of {decomposition_results.get('period', 'N/A')}). This means a simple ARIMA model is not the best fit."
            )
            next_steps = (
                "The **best case** for this data is likely a **SARIMA(p,d,q)(P,D,Q)m model**. "
                "Use the ACF/PACF plots to find (p,q) and the seasonal (P,Q) parameters. "
                "The 'm' parameter would be {decomposition_results.get('period', 'N/A')}."
            )

        elif not is_stationary and not is_decomposed:
            # Case 3: Non-Stationary, No Seasonality (or non-datetime index)
            title = "Recommendation: Tune ARIMA(p,d,q) Model"
            recommendation = (
                "The ADF test shows your data is **non-stationary (d>0)**. "
                "The decomposition plot was skipped (likely due to a non-datetime index or short series), "
                "so a standard ARIMA model is the correct approach."
            )
            if has_arima_residuals:
                next_steps = (
                    "Our **ARIMA(1,1,1) trial** is a good starting point. Now, look at the **ACF/PACF plots** to find better 'p' and 'q' values. "
                    "Check the 'Residual ACF Plot': if there are still large spikes, it means the (1,1,1) model is not perfect and can be improved by tuning 'p' or 'q'."
                )
            else:
                next_steps = "Use the ACF/PACF plots to determine the (p,d,q) orders for an ARIMA model."
        
        else:
            # Fallback
            title = "Summary"
            recommendation = "Analysis complete. Review the plots to determine the best model."
            next_steps = "Check for stationarity, seasonality, and use the ACF/PACF plots to guide your model selection."

        return {
            "title": title,
            "recommendation": recommendation,
            "next_steps": next_steps
        }
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {"error": str(e)}

# ===--- API ENDPOINTS ---===

@app.route("/inspect-csv", methods=['POST'])
def inspect_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file, nrows=5)
        return jsonify({
            "filename": file.filename,
            "columns": list(df.columns),
            "sample_data": df.head().to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": f"Error processing CSV file: {e}"}), 400

@app.route("/analyze", methods=['POST'])
def analyze_time_series():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    date_column = request.form.get('date_column')
    value_column = request.form.get('value_column')

    if not date_column or not value_column:
        return jsonify({"error": "Missing date_column or value_column"}), 400

    try:
        # We need to read the file into memory to use file.seek(0) later
        file_buffer = io.BytesIO(file.read())
        df = pd.read_csv(file_buffer)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV file: {e}"}), 400

    if date_column not in df.columns or value_column not in df.columns:
        return jsonify({"error": f"One or both columns ('{date_column}', '{value_column}') not found."}), 404

    try:
        # --- NEW CLEANING LOGIC ---
        
        # 1. Clean Value Column (e.g., "sales")
        df[value_column] = df[value_column].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        
        # 2. Clean Date Column (e.g., "mount")
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format='mixed')
        
        is_datetime_index = True
        
        # 3. Check if date conversion failed (e.g., >95% are NaT)
        if df[date_column].isnull().sum() >= len(df) * 0.95:
            print("Warning: Date conversion failed. Falling back to simple index.")
            is_datetime_index = False
            
            # Reread the file to get the original string dates
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer)
            
            # Just clean the value column again
            df[value_column] = df[value_column].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            
            original_rows = len(df)
            df.dropna(subset=[value_column], inplace=True) # Only drop if value is bad
            cleaned_rows = len(df)
        else:
            print("Info: Date conversion successful.")
            is_datetime_index = True
            original_rows = len(df)
            df.dropna(subset=[date_column, value_column], inplace=True)
            cleaned_rows = len(df)
        
        # --- END NEW CLEANING LOGIC ---
        
        if cleaned_rows == 0:
            return jsonify({"error": f"Analysis failed. After cleaning, 0 valid data rows were found. Check if '{date_column}' contains valid dates/text or if '{value_column}' contains valid numbers."}), 400
        
        # --- NEW: Check for duplicate dates BEFORE setting index ---
        if is_datetime_index and df[date_column].duplicated().any():
            return jsonify({"error": f"Analysis failed. Your date column '{date_column}' contains duplicate dates. Please aggregate or clean the file and try again."}), 400
            
        df.set_index(date_column, inplace=True)
        
        ts_interpolated = None
        freq = 'D' # Default
        
        if is_datetime_index:
            df.sort_index(inplace=True)
            ts = df[value_column]
            
            # --- NEW: Check for massive date gaps (MemoryError prevention) ---
            try:
                inferred_freq = pd.infer_freq(ts.index)
                freq = inferred_freq or 'D'
                # Limit the number of rows to 1 million to prevent memory crashes
                # This checks if resampling would create a huge dataframe
                date_range_days = (ts.index.max() - ts.index.min()).days
                if freq == 'D' and date_range_days > 1_000_000:
                     raise MemoryError(f"Date range is too large ({date_range_days} days) to interpolate. Analysis skipped.")

                ts_interpolated = ts.asfreq(freq)
            except (MemoryError, ValueError) as e:
                 print(f"Warning: Interpolation failed. {e}. Continuing without interpolation.")
                 # Fallback: just use the raw time series, skip interpolation
                 is_datetime_index = False # Treat it as a simple index from now on
                 ts_interpolated = ts.copy()
                 freq = 'unknown'

            if is_datetime_index: # If interpolation succeeded
                interpolation_method = 'time'
                if not isinstance(ts_interpolated.index, pd.DatetimeIndex):
                     interpolation_method = 'linear'
                if ts_interpolated.isna().any():
                    ts_interpolated = ts_interpolated.interpolate(method=interpolation_method)
                    ts_interpolated.bfill(inplace=True)
        else:
            ts = df[value_column]
            ts_interpolated = ts.copy() # Use the raw series
            freq = 'unknown' # Set a flag
            
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": f"Error processing columns: {e}. Check your column data types."}), 400

    # --- 4. Perform Base Analysis ---
    try:
        original_mean = float(ts.mean())
        original_variance = float(ts.var())
        original_plot_b64 = create_plot_base64(ts, f"Original Time Series: {value_column}", xlabel=date_column, ylabel=value_column)
        lag_plot_b64 = create_lag_plot_base64(ts)
        stationarity_test = check_stationarity(ts)
        
        rolling_stats_plot_b64 = None
        if is_datetime_index:
            rolling_stats_plot_b64 = create_rolling_stats_plot(ts)
            
    except Exception as e:
        return jsonify({"error": f"Error during base analysis: {e}"}), 500

    # --- 5. Handle Non-Stationary Data ---
    stationary_results = None
    if not stationarity_test.get("is_stationary", False):
        try:
            ts_stationary = ts.diff().dropna()
            if ts_stationary.empty:
                stationary_results = {"error": "Differencing resulted in empty series."}
            else:
                stationary_mean = float(ts_stationary.mean())
                stationary_variance = float(ts_stationary.var())
                stationary_plot_b64 = create_plot_base64(
                    ts_stationary, 
                    f"Stationary Time Series (1st Order Differencing)",
                    xlabel=date_column, 
                    ylabel=value_column
                )
                new_stationarity_test = check_stationarity(ts_stationary)
                stationary_results = {
                    "method": "First-Order Differencing",
                    "mean": stationary_mean,
                    "variance": stationary_variance,
                    "plot_base64": stationary_plot_b64,
                    "stationarity_test": new_stationarity_test
                }
        except Exception as e:
            stationary_results = {"error": str(e)}

    # --- 6. Perform Decomposition Analysis ---
    decomposition_results = {"error": "Skipped. Requires a valid datetime index."}
    if is_datetime_index:
        decomposition_results = run_decomposition_analysis(ts_interpolated.copy())

    # --- 7. Perform Simple ARIMA Analysis ---
    arima_results = run_simple_arima_analysis(ts_interpolated.copy(), freq, is_datetime_index)
    
    # --- 8. Generate Final Summary ---
    final_summary = generate_final_summary(
        stationarity_test, 
        decomposition_results, 
        arima_results
    )
        
    # --- 9. Assemble Final Response ---
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
            "lag_plot_base64": lag_plot_b64,
            "rolling_stats_plot_base64": rolling_stats_plot_b64, # Will be null if non-datetime
            "stationarity_test": stationarity_test
        },
        "stationary_results": stationary_results,
        "decomposition_results": decomposition_results,
        "arima_results": arima_results,
        "future_forecast_results": None,
        "final_summary": final_summary
    }
    
    return jsonify(response_data)


# This block allows you to run the app directly with `python app.py`
if __name__ == "__main__":
    app.run(debug=True)

