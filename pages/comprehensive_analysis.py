import streamlit as st
import pandas as pd
import json
from datetime import datetime
import traceback
from typing import Dict, List
import plotly.express as px

# Import the analysis functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.plotting.plot_generator import generate_comprehensive_analysis, create_comprehensive_visualizations
from src.agents.sql_agent import SQLAgent

def main():
    """Main function for comprehensive analysis page"""
    
    # Set page config
    st.set_page_config(
        page_title="Comprehensive Data Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for colorful styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .analysis-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.2);
    }
    .insight-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.2);
        border-left: 4px solid #ffffff;
    }
    .plot-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .tab-content {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #667eea;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if we have data and query from session state
    if 'data' not in st.session_state:
        st.error("❌ No data available. Please go back to the main page and upload a dataset first.")
        st.stop()
    
    if 'comprehensive_analysis_query' not in st.session_state:
        st.error("❌ No analysis query found. Please go back to the main page and start a comprehensive analysis.")
        st.stop()
    
    data = st.session_state['data']
    user_query = st.session_state['comprehensive_analysis_query']
    
    # Check if multiple files were uploaded
    uploaded_files = st.session_state.get('uploaded_files', [])
    current_file = st.session_state.get('current_file', 'Unknown')
    
    # Create dataset description
    if len(uploaded_files) > 1:
        dataset_desc = f"Combined dataset from {len(uploaded_files)} files: {', '.join(uploaded_files[:3])}"
        if len(uploaded_files) > 3:
            dataset_desc += f" and {len(uploaded_files) - 3} more"
    else:
        dataset_desc = f"Single file: {current_file}"
    
    # Header with gradient background
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Comprehensive Data Analysis</h1>
        <p>Advanced statistical insights and dynamic visualizations powered by AI</p>
        <p><strong>Query:</strong> {user_query}</p>
        <p><strong>Dataset:</strong> {data.shape[0]} rows × {data.shape[1]} columns</p>
        <p><strong>Source:</strong> {dataset_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize LLM agent for Gemini insights
    try:
        # Get agent configuration from session state
        agent_type = st.session_state.get('agent_type', 'gemini')
        model = st.session_state.get('model', 'gemini-2.0-flash')
        api_key = st.session_state.get('api_key')
        base_url = st.session_state.get('base_url')
        
        agent = SQLAgent(agent_type, model_name=model, base_url=base_url, api_key=api_key)
    except Exception as e:
        st.error(f"❌ Error initializing LLM agent: {str(e)}")
        agent = None
    
    # Generate comprehensive analysis
    with st.spinner("🔬 Performing comprehensive statistical analysis..."):
        try:
            analysis_results = generate_comprehensive_analysis(data, user_query)
            
            # Create enhanced visualizations
            visualizations = create_comprehensive_visualizations(data, analysis_results)
            
            # Display summary statistics in colorful cards
            st.markdown("### 📈 Dataset Overview")
            summary_stats = analysis_results.get('summary_stats', {})
            if 'dataset_info' in summary_stats:
                info = summary_stats['dataset_info']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📊 Rows</h3>
                        <h2>{info.get('rows', 0):,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📋 Columns</h3>
                        <h2>{info.get('columns', 0)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💾 Memory</h3>
                        <h2>{info.get('memory_usage', 0) / 1024 / 1024:.2f} MB</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔄 Duplicates</h3>
                        <h2>{info.get('duplicate_rows', 0)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Data quality analysis
            st.markdown("### 🔍 Data Quality Assessment")
            quality_issues = analysis_results.get('data_quality', {})
            
            col1, col2 = st.columns(2)
            with col1:
                if quality_issues.get('missing_values'):
                    st.markdown("""
                    <div class="analysis-section">
                        <h4>⚠️ Missing Values</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for col, info in quality_issues['missing_values'].items():
                        st.markdown(f"""
                        <div class="insight-box">
                            <strong>{col}:</strong> {info['count']} ({info['percentage']:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if quality_issues.get('outliers'):
                    st.markdown("""
                    <div class="analysis-section">
                        <h4>🎯 Outliers Detected</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for col, info in quality_issues['outliers'].items():
                        st.markdown(f"""
                        <div class="insight-box">
                            <strong>{col}:</strong> {info['count']} ({info['percentage']:.1f}%)
                        </div>
                        """, unsafe_allow_html=True)
            
            # Generate Gemini-powered conclusions
            if agent:
                st.markdown("### 🧠 AI-Powered Insights")
                conclusions = generate_gemini_conclusions(analysis_results, user_query, agent)
                
                for i, conclusion in enumerate(conclusions):
                    st.markdown(f"""
                    <div class="insight-box">
                        <h4>💡 Insight {i+1}</h4>
                        <p>{conclusion}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Dynamic plots section
            st.markdown("### 📊 Dynamic Visualizations")
            
            if visualizations:
                # Create tabs for different types of visualizations
                tab_names = ["📈 Univariate", "🔗 Bivariate", "🌐 Multivariate", "📊 Correlations"]
                tabs = st.tabs(tab_names)
                
                univariate_plots = [v for v in visualizations if v['type'] in ['histogram', 'box', 'bar']]
                bivariate_plots = [v for v in visualizations if v['type'] == 'scatter']
                multivariate_plots = [v for v in visualizations if v['type'] in ['line', 'heatmap']]
                correlation_plots = [v for v in visualizations if 'correlation' in v['title'].lower()]
                
                with tabs[0]:
                    if univariate_plots:
                        st.markdown("#### 📈 Distribution Analysis")
                        for i, plot in enumerate(univariate_plots[:4]):
                            with st.container():
                                st.markdown(f"""
                                <div class="plot-container">
                                    <h5>{plot['title']}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(plot['figure'], use_container_width=True)
                                st.caption(plot['description'])
                    else:
                        st.info("No univariate plots available.")
                
                with tabs[1]:
                    if bivariate_plots:
                        st.markdown("#### 🔗 Relationship Analysis")
                        for i, plot in enumerate(bivariate_plots[:4]):
                            with st.container():
                                st.markdown(f"""
                                <div class="plot-container">
                                    <h5>{plot['title']}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(plot['figure'], use_container_width=True)
                                st.caption(plot['description'])
                    else:
                        st.info("No bivariate plots available.")
                
                with tabs[2]:
                    if multivariate_plots:
                        st.markdown("#### 🌐 Multidimensional Analysis")
                        for i, plot in enumerate(multivariate_plots[:2]):
                            with st.container():
                                st.markdown(f"""
                                <div class="plot-container">
                                    <h5>{plot['title']}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(plot['figure'], use_container_width=True)
                                st.caption(plot['description'])
                    else:
                        st.info("No multivariate plots available.")
                
                with tabs[3]:
                    if correlation_plots:
                        st.markdown("#### 📊 Correlation Matrix")
                        for i, plot in enumerate(correlation_plots):
                            with st.container():
                                st.markdown(f"""
                                <div class="plot-container">
                                    <h5>{plot['title']}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(plot['figure'], use_container_width=True)
                                st.caption(plot['description'])
                    else:
                        st.info("No correlation plots available.")
            
            # Detailed statistical analysis
            st.markdown("### 📊 Detailed Statistical Analysis")
            
            # Univariate analysis
            with st.expander("📈 Univariate Statistics", expanded=False):
                univariate_results = analysis_results.get('univariate', {})
                for col, analysis in univariate_results.items():
                    if analysis['type'] == 'numerical':
                        stats = analysis['basic_stats']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean", f"{stats['mean']:.2f}")
                        with col2:
                            st.metric("Median", f"{stats['median']:.2f}")
                        with col3:
                            st.metric("Std Dev", f"{stats['std']:.2f}")
                        with col4:
                            st.metric("IQR", f"{stats['iqr']:.2f}")
                        st.write(f"**Column:** {col}")
                        st.divider()
                    else:
                        stats = analysis['basic_stats']
                        st.write(f"**{col}** (Categorical)")
                        st.write(f"• Unique values: {stats['unique_values']}")
                        st.write(f"• Most common: {stats['most_common']} ({stats['most_common_count']} times)")
                        st.divider()
            
            # Correlation analysis
            with st.expander("🔗 Correlation Analysis", expanded=False):
                corr_analysis = analysis_results.get('correlation_analysis', {})
                if 'strong_correlations' in corr_analysis:
                    strong_corrs = corr_analysis['strong_correlations']
                    if strong_corrs:
                        st.write("**Strong Correlations (>0.7):**")
                        for corr in strong_corrs:
                            cols = corr['columns']
                            pearson = corr['pearson']
                            st.write(f"• {cols[0]} ↔ {cols[1]}: r={pearson:.3f} ({corr['strength']})")
                    else:
                        st.write("No strong correlations found.")
            
            # Add download button for analysis results
            st.markdown("### 💾 Export Analysis")
            analysis_json = json.dumps(analysis_results, indent=2, default=str)
            st.download_button(
                label="📥 Download Analysis Results (JSON)",
                data=analysis_json,
                file_name=f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Add back button
            st.markdown("---")
            if st.button("🔙 Back to Main Page"):
                st.switch_page("app")
            
        except Exception as e:
            st.error(f"❌ Error during comprehensive analysis: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.error(traceback.format_exc())


def generate_gemini_conclusions(analysis_results: Dict, user_query: str, agent) -> List[str]:
    """Generate AI-powered conclusions using Gemini API"""
    try:
        # Prepare analysis summary for Gemini
        summary_parts = []
        
        # Dataset info
        if 'summary_stats' in analysis_results:
            info = analysis_results['summary_stats'].get('dataset_info', {})
            summary_parts.append(f"Dataset: {info.get('rows', 0)} rows, {info.get('columns', 0)} columns")
        
        # Data quality
        quality_issues = analysis_results.get('data_quality', {})
        if quality_issues.get('missing_values'):
            missing_count = len(quality_issues['missing_values'])
            summary_parts.append(f"Missing values in {missing_count} columns")
        
        if quality_issues.get('outliers'):
            outlier_count = len(quality_issues['outliers'])
            summary_parts.append(f"Outliers detected in {outlier_count} columns")
        
        # Univariate insights
        univariate_results = analysis_results.get('univariate', {})
        numerical_cols = [col for col, analysis in univariate_results.items() if analysis['type'] == 'numerical']
        categorical_cols = [col for col, analysis in univariate_results.items() if analysis['type'] == 'categorical']
        
        if numerical_cols:
            summary_parts.append(f"Numerical variables: {len(numerical_cols)}")
        if categorical_cols:
            summary_parts.append(f"Categorical variables: {len(categorical_cols)}")
        
        # Correlation insights
        corr_analysis = analysis_results.get('correlation_analysis', {})
        strong_corrs = corr_analysis.get('strong_correlations', [])
        if strong_corrs:
            summary_parts.append(f"Strong correlations: {len(strong_corrs)} found")
        
        # Create prompt for Gemini
        prompt = f"""
You are an expert data analyst. Based on the following analysis results, provide 3-4 concise, actionable insights (2-3 sentences each) that would be valuable for business decision-making.

USER QUERY: {user_query}

ANALYSIS SUMMARY:
{chr(10).join(summary_parts)}

Please provide insights that are:
1. Specific and actionable
2. Business-relevant
3. Based on the data patterns
4. Written in clear, professional language

Respond with exactly 3-4 insights, each separated by a newline. Keep each insight under 100 words.
"""
        
        # Get response from Gemini
        response = agent.query(prompt).strip()
        
        # Parse the response into individual insights
        insights = [insight.strip() for insight in response.split('\n') if insight.strip()]
        
        # Ensure we have 3-4 insights
        if len(insights) < 3:
            # Add default insights if not enough
            default_insights = [
                "The dataset shows good data quality with minimal missing values.",
                "Consider investigating outliers for potential data quality issues.",
                "Explore correlations between variables for deeper insights."
            ]
            insights.extend(default_insights[:3-len(insights)])
        elif len(insights) > 4:
            insights = insights[:4]
        
        return insights
        
    except Exception as e:
        # Fallback insights
        return [
            "The comprehensive analysis reveals important patterns in your data.",
            "Consider exploring correlations between variables for deeper insights.",
            "Data quality assessment shows areas that may need attention."
        ]


if __name__ == "__main__":
    main()
