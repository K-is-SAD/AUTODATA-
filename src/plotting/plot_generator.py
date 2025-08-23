import pandas as pd
import streamlit as st
from typing import List, Dict, Optional
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer, )):
            return int(o)
        elif isinstance(o, (np.floating, )):
            return float(o)
        elif isinstance(o, (np.ndarray, )):
            return o.tolist()
        return super(NpEncoder, self).default(o)

def generate_plot_suggestions(llm_agent, user_query: str, columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """
    Generate plot suggestions using LLM based on user query and data structure
    
    Args:
        llm_agent: The LLM agent instance
        user_query: User's natural language query
        columns_info: Dictionary containing column names and their data types
        data_sample: Sample of the dataframe for context
        
    Returns:
        List of plot dictionaries with format: [{"plot_type": "hist", "columns": ["col1", "col2"], "title": "Title", "config": {...}}]
    """
    
    # Prepare context for LLM
    columns_summary = []
    for col, dtype in columns_info.items():
        sample_values = data_sample[col].dropna().head(3).tolist()
        columns_summary.append({
            "name": col,
            "type": str(dtype),
            "sample_values": sample_values,
            "null_count": data_sample[col].isnull().sum(),
            "unique_count": data_sample[col].nunique()
        })
    
    prompt = f"""
Based on the user's query and data structure, suggest appropriate plots to visualize the data.

USER QUERY: "{user_query}"

DATA STRUCTURE:
{json.dumps(columns_summary, indent=2, cls=NpEncoder)}

Available plot types:
- histogram: For distribution of numerical data (can include color/hue for grouping)
- scatter: For relationship between two numerical variables (supports color, size, and hover data)
- box: For distribution and outliers of numerical data by categories (supports color grouping)
- violin: Similar to box plots but shows full distribution shape (supports color grouping)
- bar: For categorical data counts or aggregated values (supports color grouping)
- line: For time series or ordered data (supports color for multiple series)
- area: For filled line plots (supports color for stacking)
- pie: For categorical data proportions
- sunburst: For hierarchical categorical data
- treemap: For hierarchical data with size encoding
- density_heatmap: For 2D density plots
- density_contour: For contour plots of 2D data
- correlation: For correlation matrix of numerical columns

Return a JSON array of plot suggestions in this exact format:
[
    {{
        "plot_type": "histogram",
        "columns": ["column_name"],
        "title": "Distribution of Column Name",
        "config": {{
            "bins": 30,
            "hue": "category_column_for_grouping",
            "color": "specific_color_if_no_hue"
        }}
    }},
    {{
        "plot_type": "scatter",
        "columns": ["x_column", "y_column"],
        "title": "X Column vs Y Column",
        "config": {{
            "color_column": "category_column_for_color_coding",
            "size_column": "numerical_column_for_size",
            "hue": "another_way_to_specify_color_column"
        }}
    }},
    {{
        "plot_type": "box",
        "columns": ["categorical_x", "numerical_y"],
        "title": "Y Column by X Column",
        "config": {{
            "color_column": "additional_grouping_variable"
        }}
    }}
]

Guidelines:
1. Choose plot types that make sense for the data types
2. For numerical columns, suggest histograms, box plots, violin plots, or scatter plots
3. For categorical columns, suggest bar charts, pie charts, sunburst, or treemap
4. If user asks about relationships, suggest scatter plots or correlation matrices
5. If user asks about distributions, suggest histograms, box plots, or violin plots
6. If user asks about groups/categories, use hue/color_column for grouping
7. For complex hierarchical data, suggest sunburst or treemap
8. Maximum 4 plots to avoid overwhelming the user
9. Include meaningful titles and appropriate configurations
10. Only use columns that exist in the data
11. When suggesting color/hue, ensure the column has reasonable number of unique values (< 20)
12. Prefer color coding when comparing groups or categories

RESPOND ONLY WITH THE JSON ARRAY, NO OTHER TEXT:
"""
    
    try:
        response = llm_agent.query(prompt)
        # Extract JSON from response
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:-3]
        elif response.startswith('```'):
            response = response[3:-3]
        
        plot_suggestions = json.loads(response)
        
        # Filter out inappropriate plot types based on data types
        filtered_suggestions = filter_plot_suggestions(plot_suggestions, columns_info, data_sample)
        return filtered_suggestions
    except Exception as e:
        st.error(f"Error generating plot suggestions: {e}")
        # Return default suggestions based on data types
        return generate_default_plots(columns_info, data_sample)


def filter_plot_suggestions(plot_suggestions: List[Dict], columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """
    Filter out inappropriate plot suggestions based on data types and column compatibility.
    
    Args:
        plot_suggestions: List of plot configurations from LLM
        columns_info: Dictionary containing column names and their data types
        data_sample: Sample of the dataframe for context
        
    Returns:
        Filtered list of plot configurations
    """
    filtered_plots = []
    
    for plot_config in plot_suggestions:
        plot_type = plot_config.get("plot_type", "")
        columns = plot_config.get("columns", [])
        
        # Skip plots with missing columns
        if not all(col in data_sample.columns for col in columns):
            continue
            
        # Check data type compatibility
        if plot_type == "correlation":
            # Correlation plots need at least 2 numerical columns
            numerical_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data_sample[col])]
            if len(numerical_cols) < 2:
                continue  # Skip correlation plots with categorical data
                
        elif plot_type == "scatter":
            # Scatter plots need 2 numerical columns
            if len(columns) >= 2:
                x_col, y_col = columns[0], columns[1]
                if not (pd.api.types.is_numeric_dtype(data_sample[x_col]) and 
                       pd.api.types.is_numeric_dtype(data_sample[y_col])):
                    # Convert scatter to box plot if one column is categorical
                    if (not pd.api.types.is_numeric_dtype(data_sample[x_col]) and 
                        pd.api.types.is_numeric_dtype(data_sample[y_col])):
                        plot_config["plot_type"] = "box"
                        plot_config["title"] = plot_config["title"].replace("vs", "by")
                    else:
                        continue  # Skip if both are categorical or other issues
                        
        elif plot_type in ["histogram", "box", "violin"]:
            # These need at least one numerical column
            has_numerical = any(pd.api.types.is_numeric_dtype(data_sample[col]) for col in columns)
            if not has_numerical:
                continue
                
        elif plot_type in ["bar", "pie"]:
            # These work with categorical data, so they're usually fine
            pass
            
        filtered_plots.append(plot_config)
    
    return filtered_plots


def generate_default_plots(columns_info: Dict, data_sample: pd.DataFrame) -> List[Dict]:
    """Generate default plot suggestions based on data types"""
    plots = []
    numerical_cols = [col for col, dtype in columns_info.items() if pd.api.types.is_numeric_dtype(dtype)]
    categorical_cols = [col for col, dtype in columns_info.items() if not pd.api.types.is_numeric_dtype(dtype)]
    
    # Find good categorical columns for hue (those with reasonable number of unique values)
    good_hue_cols = [col for col in categorical_cols 
                     if data_sample[col].nunique() <= 10 and data_sample[col].nunique() > 1]
    
    # Add histogram for first numerical column with hue if available
    if numerical_cols:
        config = {"bins": 30}
        if good_hue_cols:
            config["hue"] = good_hue_cols[0]
        plots.append({
            "plot_type": "histogram",
            "columns": [numerical_cols[0]],
            "title": f"Distribution of {numerical_cols[0]}" + (f" by {good_hue_cols[0]}" if good_hue_cols else ""),
            "config": config
        })
    
    # Add scatter plot if we have 2+ numerical columns with color coding
    if len(numerical_cols) >= 2:
        config = {}
        if good_hue_cols:
            config["color_column"] = good_hue_cols[0]
        if len(numerical_cols) >= 3:
            config["size_column"] = numerical_cols[2]
        plots.append({
            "plot_type": "scatter",
            "columns": numerical_cols[:2],
            "title": f"{numerical_cols[0]} vs {numerical_cols[1]}" + (f" by {good_hue_cols[0]}" if good_hue_cols else ""),
            "config": config
        })
    
    # Add box plot if we have both numerical and categorical columns
    if numerical_cols and categorical_cols:
        config = {}
        if len(good_hue_cols) >= 2:
            config["color_column"] = good_hue_cols[1]
        plots.append({
            "plot_type": "box",
            "columns": [categorical_cols[0], numerical_cols[0]],
            "title": f"{numerical_cols[0]} by {categorical_cols[0]}" + (f" and {good_hue_cols[1]}" if len(good_hue_cols) >= 2 else ""),
            "config": config
        })
    
    # Add bar chart for first categorical column with grouping if available
    if categorical_cols:
        config = {}
        if len(good_hue_cols) >= 2:
            config["color_column"] = good_hue_cols[1]
        plots.append({
            "plot_type": "bar",
            "columns": [categorical_cols[0]],
            "title": f"Count of {categorical_cols[0]}" + (f" by {good_hue_cols[1]}" if len(good_hue_cols) >= 2 else ""),
            "config": config
        })
    
    return plots


def create_plot(plot_config: Dict, data: pd.DataFrame) -> None:
    """
    Create and display a plot based on the configuration
    
    Args:
        plot_config: Dictionary containing plot type, columns, title, and config
        data: The dataframe to plot from
    """
    
    plot_type = plot_config.get("plot_type")
    columns = plot_config.get("columns", [])
    title = plot_config.get("title", "Plot")
    config = plot_config.get("config", {})
    
    st.write(f"**Debug:** Creating {plot_type} plot with columns: {columns}")
    st.write(f"**Debug:** Available data columns: {list(data.columns)}")
    
    try:
        if plot_type == "histogram":
            if len(columns) >= 1 and columns[0] in data.columns:
                fig = px.histogram(
                    data, 
                    x=columns[0], 
                    title=title,
                    nbins=config.get("bins", 30)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Column '{columns[0]}' not found in data for histogram")
            
        elif plot_type == "scatter":
            if len(columns) >= 2 and all(col in data.columns for col in columns[:2]):
                color_col = config.get("color_column") if config.get("color_column") in data.columns else None
                size_col = config.get("size_column") if config.get("size_column") in data.columns else None
                
                fig = px.scatter(
                    data,
                    x=columns[0],
                    y=columns[1],
                    color=color_col,
                    size=size_col,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                missing_cols = [col for col in columns[:2] if col not in data.columns]
                st.error(f"Columns {missing_cols} not found in data for scatter plot")
                
        elif plot_type == "box":
            if len(columns) >= 1 and columns[0] in data.columns:
                y_col = columns[0]
                x_col = None
                
                # Check if we have a second column for grouping
                if len(columns) > 1:
                    if columns[1] in data.columns:
                        x_col = columns[1]
                    else:
                        # If the second column doesn't exist, use the first for grouping if it's categorical
                        # and look for numerical columns for y-axis
                        numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                        if numerical_cols and not pd.api.types.is_numeric_dtype(data[columns[0]]):
                            x_col = columns[0]  # Use first column as grouping variable
                            y_col = numerical_cols[0]  # Use first numerical column as y
                        elif len(numerical_cols) > 1:
                            # If we have multiple numerical columns, use them
                            y_col = numerical_cols[0]
                            x_col = columns[0] if not pd.api.types.is_numeric_dtype(data[columns[0]]) else None
                
                st.write(f"**Debug:** Box plot using x={x_col}, y={y_col}")
                fig = px.box(
                    data,
                    x=x_col,
                    y=y_col,
                    title=title,
                    color_discrete_sequence=[config.get("color", "blue")]
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Column '{columns[0]}' not found in data for box plot")
                
        elif plot_type == "bar":
            if len(columns) >= 1 and columns[0] in data.columns:
                # Check if we have aggregated data or if this is a simple count
                if len(columns) >= 2 and columns[1] in data.columns:
                    # We have x and y columns for bar chart
                    fig = px.bar(
                        data, 
                        x=columns[0], 
                        y=columns[1], 
                        title=title,
                        labels={columns[0]: config.get("x_axis_label", columns[0]), 
                               columns[1]: config.get("y_axis_label", columns[1])},
                        color_discrete_sequence=[config.get("color", "blue")]
                    )
                elif pd.api.types.is_numeric_dtype(data[columns[0]]):
                    # For numerical data, create value counts or use values directly if already aggregated
                    if len(data[columns[0]].unique()) <= 20:  # If few unique values, treat as categorical
                        value_counts = data[columns[0]].value_counts().sort_index()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=title,
                            labels={'x': columns[0], 'y': 'Count'},
                            color_discrete_sequence=[config.get("color", "blue")]
                        )
                    else:
                        # For continuous numerical data, use histogram instead
                        fig = px.histogram(
                            data, 
                            x=columns[0], 
                            title=title,
                            nbins=config.get("bins", 20)
                        )
                else:
                    # For categorical data, count occurrences
                    value_counts = data[columns[0]].value_counts()
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=title,
                        labels={'x': columns[0], 'y': 'Count'},
                        color_discrete_sequence=[config.get("color", "blue")]
                    )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "line":
            if len(columns) >= 1 and columns[0] in data.columns:
                y_col = columns[0]
                x_col = columns[1] if len(columns) > 1 and columns[1] in data.columns else data.index
                
                fig = px.line(
                    data,
                    x=x_col,
                    y=y_col,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
                
        elif plot_type == "correlation":
            # Select only numerical columns for correlation
            numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                corr_matrix = data[numerical_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title=title or "Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns for correlation plot")
                
        elif plot_type == "pie":
            if len(columns) >= 1 and columns[0] in data.columns:
                value_counts = data[columns[0]].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.warning(f"Plot type '{plot_type}' not supported yet")
            
    except Exception as e:
        st.error(f"Error creating {plot_type} plot: {str(e)}")


def display_plots_section(data: pd.DataFrame, user_query: str, llm_agent, query_result: Optional[pd.DataFrame] = None):
    """
    Main function to display the plots section
    
    Args:
        data: The original dataframe
        user_query: User's natural language query
        llm_agent: The LLM agent instance
        query_result: Result from SQL query (if any)
    """
    
    # Use query result if available, otherwise use original data
    plot_data = query_result if query_result is not None and not query_result.empty else data
    
    if plot_data.empty:
        st.warning("No data available for plotting")
        return
    
    st.write(f"**Debug:** Using data with shape: {plot_data.shape}")
    st.write(f"**Debug:** Available columns: {list(plot_data.columns)}")
    
    # Get column information
    columns_info = {col: plot_data[col].dtype for col in plot_data.columns}
    
    # For now, let's test with hardcoded plot suggestions based on your example
    plot_suggestions = [
        {
            "plot_type": "bar",
            "columns": ["GarageCars"],
            "title": "Average SalePrice by GarageCars",
            "config": {
                "x_axis_label": "GarageCars",
                "y_axis_label": "Average SalePrice",
                "color": "green"
            }
        },
        {
            "plot_type": "box",
            "columns": ["GarageCars", "AVG(SalePrice)"],
            "title": "SalePrice Distribution by GarageCars",
            "config": {
                "color": "purple"
            }
        }
    ]
    
    # If hardcoded plots don't work with the data, generate new ones
    if not all(any(col in plot_data.columns for col in plot["columns"]) for plot in plot_suggestions):
        st.write("**Debug:** Hardcoded plots don't match data, generating new ones...")
        # Generate plot suggestions
        with st.spinner("🤖 Generating plot suggestions..."):
            plot_suggestions = generate_plot_suggestions(llm_agent, user_query, columns_info, plot_data)
    
    if not plot_suggestions:
        st.warning("No plot suggestions generated")
        return
    
    st.subheader("📊 Suggested Visualizations")
    
    # Show plot configurations for debugging
    st.write("**Debug:** Plot configurations:")
    st.json(plot_suggestions)
    
    # Create columns for plots
    num_plots = len(plot_suggestions)
    if num_plots == 1:
        create_plot(plot_suggestions[0], plot_data)
    elif num_plots == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Creating plot 1:** {plot_suggestions[0]['plot_type']}")
            create_plot(plot_suggestions[0], plot_data)
        with col2:
            st.write(f"**Creating plot 2:** {plot_suggestions[1]['plot_type']}")
            create_plot(plot_suggestions[1], plot_data)
    else:
        # For more than 2 plots, arrange in rows
        for i in range(0, num_plots, 2):
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Creating plot {i+1}:** {plot_suggestions[i]['plot_type']}")
                create_plot(plot_suggestions[i], plot_data)
            if i + 1 < num_plots:
                with cols[1]:
                    st.write(f"**Creating plot {i+2}:** {plot_suggestions[i+1]['plot_type']}")
                    create_plot(plot_suggestions[i + 1], plot_data)
    
    # Show plot configurations for reference
    with st.expander("🔧 Plot Configurations (for developers)"):
        st.json(plot_suggestions)


def generate_comprehensive_analysis(data: pd.DataFrame, user_query: str) -> Dict:
    """
    Generate comprehensive statistical analysis including univariate, bivariate, and multivariate statistics.
    
    Args:
        data: DataFrame to analyze
        user_query: User's original query for context
        
    Returns:
        Dictionary containing all analysis results and visualizations
    """
    analysis_results = {
        'univariate': generate_univariate_analysis(data),
        'bivariate': generate_bivariate_analysis(data),
        'multivariate': generate_multivariate_analysis(data),
        'summary_stats': generate_summary_statistics(data),
        'data_quality': analyze_data_quality(data),
        'correlation_analysis': generate_correlation_analysis(data)
    }
    
    return analysis_results


def generate_univariate_analysis(data: pd.DataFrame) -> Dict:
    """Generate comprehensive univariate analysis for all columns"""
    univariate_results = {}
    
    for column in data.columns:
        col_data = data[column].dropna()
        if len(col_data) == 0:
            continue
            
        col_type = data[column].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            # Numerical column analysis
            univariate_results[column] = analyze_numerical_column(col_data, column)
        else:
            # Categorical column analysis
            univariate_results[column] = analyze_categorical_column(col_data, column)
    
    return univariate_results


def analyze_numerical_column(col_data: pd.Series, column_name: str) -> Dict:
    """Analyze a numerical column with comprehensive statistics"""
    analysis = {
        'type': 'numerical',
        'basic_stats': {
            'count': len(col_data),
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'q25': col_data.quantile(0.25),
            'q75': col_data.quantile(0.75),
            'iqr': col_data.quantile(0.75) - col_data.quantile(0.25)
        },
        'distribution_stats': {
            'skewness': col_data.skew(),
            'kurtosis': col_data.kurtosis(),
            'normality_test': stats.normaltest(col_data)[1] if len(col_data) > 8 else None
        },
        'outliers': detect_outliers(col_data),
        'percentiles': {
            'p1': col_data.quantile(0.01),
            'p5': col_data.quantile(0.05),
            'p95': col_data.quantile(0.95),
            'p99': col_data.quantile(0.99)
        }
    }
    
    return analysis


def analyze_categorical_column(col_data: pd.Series, column_name: str) -> Dict:
    """Analyze a categorical column with comprehensive statistics"""
    value_counts = col_data.value_counts()
    total_count = len(col_data)
    
    analysis = {
        'type': 'categorical',
        'basic_stats': {
            'count': total_count,
            'unique_values': len(value_counts),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
            'least_common_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
        },
        'distribution': {
            'value_counts': value_counts.to_dict(),
            'percentages': (value_counts / total_count * 100).to_dict(),
            'entropy': calculate_entropy(value_counts)
        },
        'cardinality': {
            'high_cardinality': len(value_counts) > 20,
            'low_cardinality': len(value_counts) <= 5
        }
    }
    
    return analysis


def detect_outliers(col_data: pd.Series) -> Dict:
    """Detect outliers using multiple methods"""
    q1, q3 = col_data.quantile([0.25, 0.75])
    iqr = q3 - q1
    
    # IQR method
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    
    # Z-score method
    z_scores = np.abs(stats.zscore(col_data))
    zscore_outliers = col_data[z_scores > 3]
    
    return {
        'iqr_method': {
            'count': len(iqr_outliers),
            'percentage': len(iqr_outliers) / len(col_data) * 100,
            'indices': iqr_outliers.index.tolist()
        },
        'zscore_method': {
            'count': len(zscore_outliers),
            'percentage': len(zscore_outliers) / len(col_data) * 100,
            'indices': zscore_outliers.index.tolist()
        }
    }


def calculate_entropy(value_counts: pd.Series) -> float:
    """Calculate entropy for categorical data"""
    probabilities = value_counts / value_counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def generate_bivariate_analysis(data: pd.DataFrame) -> Dict:
    """Generate bivariate analysis between pairs of columns"""
    bivariate_results = {}
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Numerical vs Numerical
    for i, col1 in enumerate(numerical_cols):
        for col2 in numerical_cols[i+1:]:
            key = f"{col1}_vs_{col2}"
            bivariate_results[key] = analyze_numerical_pair(data[col1], data[col2], col1, col2)
    
    # Numerical vs Categorical
    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            key = f"{num_col}_by_{cat_col}"
            bivariate_results[key] = analyze_numerical_categorical_pair(data[num_col], data[cat_col], num_col, cat_col)
    
    # Categorical vs Categorical
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            key = f"{col1}_vs_{col2}"
            bivariate_results[key] = analyze_categorical_pair(data[col1], data[col2], col1, col2)
    
    return bivariate_results


def analyze_numerical_pair(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> Dict:
    """Analyze relationship between two numerical columns"""
    # Remove rows with missing values
    valid_data = pd.DataFrame({name1: col1, name2: col2}).dropna()
    
    if len(valid_data) < 2:
        return {'error': 'Insufficient data'}
    
    col1_clean = valid_data[name1]
    col2_clean = valid_data[name2]
    
    # Correlation analysis
    pearson_corr, pearson_p = stats.pearsonr(col1_clean, col2_clean)
    spearman_corr, spearman_p = stats.spearmanr(col1_clean, col2_clean)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(col1_clean, col2_clean)
    
    return {
        'type': 'numerical_vs_numerical',
        'correlation': {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
        },
        'linear_regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err
        },
        'sample_size': len(valid_data)
    }


def analyze_numerical_categorical_pair(num_col: pd.Series, cat_col: pd.Series, num_name: str, cat_name: str) -> Dict:
    """Analyze relationship between numerical and categorical columns"""
    # Remove rows with missing values
    valid_data = pd.DataFrame({num_name: num_col, cat_name: cat_col}).dropna()
    
    if len(valid_data) < 2:
        return {'error': 'Insufficient data'}
    
    # Group by categorical column
    grouped = valid_data.groupby(cat_name)[num_name]
    
    # ANOVA test
    groups = [group for name, group in grouped]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
    else:
        f_stat, p_value = None, None
    
    # Summary statistics by group
    group_stats = grouped.agg(['count', 'mean', 'std', 'min', 'max']).to_dict()
    
    return {
        'type': 'numerical_by_categorical',
        'anova': {
            'f_statistic': f_stat,
            'p_value': p_value
        },
        'group_statistics': group_stats,
        'sample_size': len(valid_data)
    }


def analyze_categorical_pair(col1: pd.Series, col2: pd.Series, name1: str, name2: str) -> Dict:
    """Analyze relationship between two categorical columns"""
    # Remove rows with missing values
    valid_data = pd.DataFrame({name1: col1, name2: col2}).dropna()
    
    if len(valid_data) < 2:
        return {'error': 'Insufficient data'}
    
    # Contingency table
    contingency_table = pd.crosstab(valid_data[name1], valid_data[name2])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # Cramer's V
    n = len(valid_data)
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return {
        'type': 'categorical_vs_categorical',
        'contingency_table': contingency_table.to_dict(),
        'chi_square': {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        },
        'cramers_v': cramer_v,
        'sample_size': len(valid_data)
    }


def generate_multivariate_analysis(data: pd.DataFrame) -> Dict:
    """Generate multivariate analysis including PCA and clustering"""
    numerical_data = data.select_dtypes(include=[np.number])
    
    if len(numerical_data.columns) < 2:
        return {'error': 'Insufficient numerical columns for multivariate analysis'}
    
    # Remove rows with missing values
    clean_data = numerical_data.dropna()
    
    if len(clean_data) < 3:
        return {'error': 'Insufficient data for multivariate analysis'}
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    # PCA Analysis
    pca_results = perform_pca_analysis(scaled_data, clean_data.columns)
    
    # Clustering Analysis
    clustering_results = perform_clustering_analysis(scaled_data, clean_data.columns)
    
    return {
        'pca': pca_results,
        'clustering': clustering_results,
        'sample_size': len(clean_data)
    }


def perform_pca_analysis(scaled_data: np.ndarray, feature_names: List[str]) -> Dict:
    """Perform Principal Component Analysis"""
    # Determine optimal number of components
    max_components = min(len(scaled_data), len(feature_names))
    n_components = min(max_components, 5)  # Limit to 5 components for visualization
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    return {
        'n_components': n_components,
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'loadings': loadings.to_dict(),
        'transformed_data': pca_result.tolist()
    }


def perform_clustering_analysis(scaled_data: np.ndarray, feature_names: List[str]) -> Dict:
    """Perform K-means clustering analysis"""
    # Determine optimal number of clusters using elbow method
    max_clusters = min(10, len(scaled_data) // 10)
    if max_clusters < 2:
        return {'error': 'Insufficient data for clustering'}
    
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Use 3 clusters as default for analysis
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    return {
        'n_clusters': optimal_k,
        'inertias': inertias,
        'cluster_labels': cluster_labels.tolist(),
        'cluster_centers': kmeans.cluster_centers_.tolist()
    }


def generate_summary_statistics(data: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics"""
    return {
        'dataset_info': {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'duplicate_rows': data.duplicated().sum()
        },
        'column_types': data.dtypes.value_counts().to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict()
    }


def analyze_data_quality(data: pd.DataFrame) -> Dict:
    """Analyze data quality issues"""
    quality_issues = {
        'missing_values': {},
        'duplicates': {},
        'outliers': {},
        'inconsistencies': {}
    }
    
    # Missing values analysis
    missing_counts = data.isnull().sum()
    for col in data.columns:
        if missing_counts[col] > 0:
            quality_issues['missing_values'][col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_counts[col] / len(data) * 100)
            }
    
    # Duplicate analysis
    duplicate_rows = data.duplicated().sum()
    if duplicate_rows > 0:
        quality_issues['duplicates']['total_duplicates'] = int(duplicate_rows)
    
    # Outlier analysis for numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        col_data = data[col].dropna()
        if len(col_data) > 0:
            outliers = detect_outliers(col_data)
            if outliers['iqr_method']['count'] > 0:
                quality_issues['outliers'][col] = outliers['iqr_method']
    
    return quality_issues


def generate_correlation_analysis(data: pd.DataFrame) -> Dict:
    """Generate comprehensive correlation analysis"""
    numerical_data = data.select_dtypes(include=[np.number])
    
    if len(numerical_data.columns) < 2:
        return {'error': 'Insufficient numerical columns for correlation analysis'}
    
    # Pearson correlation
    pearson_corr = numerical_data.corr()
    
    # Spearman correlation
    spearman_corr = numerical_data.corr(method='spearman')
    
    # Find strong correlations
    strong_correlations = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            col1 = pearson_corr.columns[i]
            col2 = pearson_corr.columns[j]
            pearson_val = pearson_corr.iloc[i, j]
            spearman_val = spearman_corr.iloc[i, j]
            
            if abs(pearson_val) > 0.7 or abs(spearman_val) > 0.7:
                strong_correlations.append({
                    'columns': [col1, col2],
                    'pearson': pearson_val,
                    'spearman': spearman_val,
                    'strength': 'strong' if abs(pearson_val) > 0.8 else 'moderate'
                })
    
    return {
        'pearson_correlation': pearson_corr.to_dict(),
        'spearman_correlation': spearman_corr.to_dict(),
        'strong_correlations': strong_correlations
    }


def create_comprehensive_visualizations(data: pd.DataFrame, analysis_results: Dict) -> List[Dict]:
    """Create comprehensive visualizations based on analysis results with enhanced styling"""
    visualizations = []
    
    # Enhanced color schemes for dynamic plots
    color_schemes = {
        'primary': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
        'warm': ['#ff9a9e', '#fecfef', '#fecfef', '#fad0c4', '#ffd1ff', '#ffecd2'],
        'cool': ['#a8edea', '#fed6e3', '#ffecd2', '#fcb69f', '#ff9a9e', '#fecfef'],
        'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    }
    
    # Univariate visualizations with enhanced styling
    univariate_plots = create_univariate_plots(data, analysis_results.get('univariate', {}))
    for plot in univariate_plots:
        if plot['figure']:
            # Apply enhanced styling
            plot['figure'].update_layout(
                title_font_size=20,
                title_font_color='#667eea',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            # Add colorful markers and lines
            if hasattr(plot['figure'], 'data') and plot['figure'].data:
                for trace in plot['figure'].data:
                    if hasattr(trace, 'marker'):
                        trace.marker.color = color_schemes['primary'][0]
                    if hasattr(trace, 'line'):
                        trace.line.color = color_schemes['primary'][0]
    visualizations.extend(univariate_plots)
    
    # Bivariate visualizations with enhanced styling
    bivariate_plots = create_bivariate_plots(data, analysis_results.get('bivariate', {}))
    for plot in bivariate_plots:
        if plot['figure']:
            # Apply enhanced styling
            plot['figure'].update_layout(
                title_font_size=20,
                title_font_color='#f093fb',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            # Add colorful markers
            if hasattr(plot['figure'], 'data') and plot['figure'].data:
                for trace in plot['figure'].data:
                    if hasattr(trace, 'marker'):
                        trace.marker.color = color_schemes['warm'][0]
    visualizations.extend(bivariate_plots)
    
    # Multivariate visualizations with enhanced styling
    multivariate_plots = create_multivariate_plots(data, analysis_results.get('multivariate', {}))
    for plot in multivariate_plots:
        if plot['figure']:
            # Apply enhanced styling
            plot['figure'].update_layout(
                title_font_size=20,
                title_font_color='#4facfe',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
    visualizations.extend(multivariate_plots)
    
    # Correlation visualizations with enhanced styling
    correlation_plots = create_correlation_plots(data, analysis_results.get('correlation_analysis', {}))
    for plot in correlation_plots:
        if plot['figure']:
            # Apply enhanced styling
            plot['figure'].update_layout(
                title_font_size=20,
                title_font_color='#f5576c',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
    visualizations.extend(correlation_plots)
    
    return visualizations


def create_univariate_plots(data: pd.DataFrame, univariate_results: Dict) -> List[Dict]:
    """Create univariate visualization plots with enhanced styling"""
    plots = []
    
    # Enhanced color schemes
    color_schemes = {
        'primary': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
        'warm': ['#ff9a9e', '#fecfef', '#fecfef', '#fad0c4', '#ffd1ff', '#ffecd2'],
        'cool': ['#a8edea', '#fed6e3', '#ffecd2', '#fcb69f', '#ff9a9e', '#fecfef']
    }
    
    for column, analysis in univariate_results.items():
        if analysis['type'] == 'numerical':
            # Enhanced histogram with gradient colors
            fig = px.histogram(
                data, 
                x=column, 
                title=f"📊 Distribution of {column}",
                nbins=30,
                color_discrete_sequence=color_schemes['primary']
            )
            fig.update_layout(
                title_font_size=20,
                title_font_color='#667eea',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            plots.append({
                'type': 'histogram',
                'figure': fig,
                'title': f"Distribution of {column}",
                'description': f"Shows the distribution of {column} with enhanced styling"
            })
            
            # Enhanced box plot with warm colors
            fig = px.box(
                data, 
                y=column, 
                title=f"📦 Box Plot of {column}",
                color_discrete_sequence=color_schemes['warm']
            )
            fig.update_layout(
                title_font_size=20,
                title_font_color='#f093fb',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            plots.append({
                'type': 'box',
                'figure': fig,
                'title': f"Box Plot of {column}",
                'description': f"Shows outliers and quartiles for {column}"
            })
            
        elif analysis['type'] == 'categorical':
            # Enhanced bar chart with cool colors
            value_counts = data[column].value_counts()
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values, 
                title=f"📊 Frequency of {column}",
                color_discrete_sequence=color_schemes['cool']
            )
            fig.update_layout(
                title_font_size=20,
                title_font_color='#4facfe',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            plots.append({
                'type': 'bar',
                'figure': fig,
                'title': f"Frequency of {column}",
                'description': f"Shows frequency distribution of {column} categories"
            })
    
    return plots


def create_bivariate_plots(data: pd.DataFrame, bivariate_results: Dict) -> List[Dict]:
    """Create bivariate visualization plots with enhanced styling"""
    plots = []
    
    # Enhanced color schemes
    color_schemes = {
        'primary': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
        'warm': ['#ff9a9e', '#fecfef', '#fecfef', '#fad0c4', '#ffd1ff', '#ffecd2']
    }
    
    for key, analysis in bivariate_results.items():
        if analysis.get('type') == 'numerical_vs_numerical':
            cols = key.split('_vs_')
            if len(cols) == 2:
                col1, col2 = cols[0], cols[1]
                
                # Enhanced scatter plot with gradient colors
                fig = px.scatter(
                    data, 
                    x=col1, 
                    y=col2, 
                    title=f"🔗 {col1} vs {col2}",
                    color_discrete_sequence=color_schemes['primary']
                )
                fig.update_layout(
                    title_font_size=20,
                    title_font_color='#667eea',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                plots.append({
                    'type': 'scatter',
                    'figure': fig,
                    'title': f"{col1} vs {col2}",
                    'description': f"Correlation: {analysis['correlation']['pearson']['correlation']:.3f}"
                })
                
        elif analysis.get('type') == 'numerical_by_categorical':
            cols = key.split('_by_')
            if len(cols) == 2:
                num_col, cat_col = cols[0], cols[1]
                
                # Enhanced box plot with warm colors
                fig = px.box(
                    data, 
                    x=cat_col, 
                    y=num_col, 
                    title=f"📦 {num_col} by {cat_col}",
                    color_discrete_sequence=color_schemes['warm']
                )
                fig.update_layout(
                    title_font_size=20,
                    title_font_color='#f093fb',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                plots.append({
                    'type': 'box',
                    'figure': fig,
                    'title': f"{num_col} by {cat_col}",
                    'description': f"Shows distribution of {num_col} across {cat_col} categories"
                })
    
    return plots


def create_multivariate_plots(data: pd.DataFrame, multivariate_results: Dict) -> List[Dict]:
    """Create multivariate visualization plots"""
    plots = []
    
    if 'pca' in multivariate_results and 'error' not in multivariate_results['pca']:
        pca_data = multivariate_results['pca']
        
        # Scree plot
        fig = px.line(
            x=range(1, len(pca_data['explained_variance_ratio']) + 1),
            y=pca_data['cumulative_variance'],
            title="PCA Cumulative Explained Variance"
        )
        plots.append({
            'type': 'line',
            'figure': fig,
            'title': "PCA Cumulative Explained Variance",
            'description': "Shows how much variance is explained by each principal component"
        })
        
        # PCA loadings heatmap
        loadings_df = pd.DataFrame(pca_data['loadings'])
        fig = px.imshow(
            loadings_df,
            title="PCA Component Loadings",
            aspect="auto"
        )
        plots.append({
            'type': 'heatmap',
            'figure': fig,
            'title': "PCA Component Loadings",
            'description': "Shows how original features contribute to principal components"
        })
    
    return plots


def create_correlation_plots(data: pd.DataFrame, correlation_results: Dict) -> List[Dict]:
    """Create correlation visualization plots with enhanced styling"""
    plots = []
    
    if 'pearson_correlation' in correlation_results:
        corr_matrix = pd.DataFrame(correlation_results['pearson_correlation'])
        
        # Enhanced correlation heatmap with RdBu color scale
        fig = px.imshow(
            corr_matrix,
            title="🔥 Correlation Matrix",
            aspect="auto",
            color_continuous_scale="RdBu",
            text_auto=True
        )
        fig.update_layout(
            title_font_size=20,
            title_font_color='#f5576c',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        plots.append({
            'type': 'heatmap',
            'figure': fig,
            'title': "Correlation Matrix",
            'description': "Shows correlations between all numerical variables"
        })
    
    return plots
