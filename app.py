import streamlit as st
import pandas as pd
import os
from datetime import datetime
import traceback
import subprocess
import logging
import json
from colorama import Fore
try:
    from src.agents.sql_agent import SQLAgent
    from src.utils.data_utils import (get_table_info, execute_sql_query,
                       save_data_info, create_sqlite_db, create_plot_figure)
    from src.agents.insight_agent import InsightAgent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.dialog("📋 Dataset Preview")
def show_data_preview(data, filename):
    st.subheader(f"Data Preview: {filename}")
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
    
    tab1, tab2 = st.tabs(["Preview", "Summary"])
    
    with tab1:
        st.dataframe(data.head(20), use_container_width=True)
    
    with tab2:
        st.write("**Data Types:**")
        dtypes_df = pd.DataFrame({
            'Column': data.dtypes.index,
            'Type': data.dtypes.values,
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values
        })
        st.dataframe(dtypes_df, use_container_width=True)
        
        st.write("**Summary Statistics:**")
        st.dataframe(data.describe(), use_container_width=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'plots_history' not in st.session_state:
        st.session_state.plots_history = []
    if 'show_query_panel' not in st.session_state:
        st.session_state.show_query_panel = True
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'smart_summary_enabled' not in st.session_state:
        st.session_state.smart_summary_enabled = True
    if 'summaries' not in st.session_state:
        st.session_state.summaries = []  # list of {timestamp, question, summary, suggestions}
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = []  # recent compact metrics texts

def add_to_chat_history(question, sql_query, result, error=None, timestamp=None):
    """Add a query and its result to chat history"""
    if timestamp is None:
        timestamp = datetime.now()
    
    chat_entry = {
        'timestamp': timestamp,
        'question': question,
        'sql_query': sql_query,
        'result': result,
        'error': error,
        'result_shape': result.shape if result is not None and not result.empty else None
    }
    st.session_state.chat_history.append(chat_entry)

def add_plot_to_history(plot_data, question, timestamp=None):
    """Add a plot to the plots history"""
    if timestamp is None:
        timestamp = datetime.now()
    
    plot_entry = {
        'timestamp': timestamp,
        'question': question,
        'plot_data': plot_data
    }
    # Add to beginning for latest-first display
    st.session_state.plots_history.insert(0, plot_entry)

def render_sidebar():
    """Render the sidebar with configuration and file upload"""
    st.sidebar.header("🤖 LLM Configuration")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.toggle(
        "Debug Mode",
        value=st.session_state.debug_mode,
        help="Show debug information and logs"
    )
    
    agent_type = st.sidebar.selectbox(
        "Choose LLM Agent:",
        ["ollama", "openai", "gemini"],
        index=0
    )
    
    # Configuration based on agent type
    base_url = None  # Initialize base_url
    if agent_type == "openai":
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        model = st.sidebar.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
    elif agent_type == "gemini":
        api_key = st.sidebar.text_input("Gemini API Key:", type="password")
        model = st.sidebar.selectbox("Model:", ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.0-flash"])
    else:  # ollama
        base_url = st.sidebar.text_input("Ollama Base URL:", value="http://localhost:11434")
        try:
            models = subprocess.run(['ollama', 'ls'], capture_output=True, text=True).stdout.strip().split('\n')
            models = [model.split()[0] for model in models if not model.split()[0].isalpha()]
            if not models:
                models = []  # fallback
        except Exception:
            models = []  # fallback
        model = st.sidebar.selectbox("Model:", models)
        api_key = None
    
    # Visualization settings
    st.sidebar.header("📊 Visualization Settings")
    enable_plots = st.sidebar.toggle(
        "Generate Plots",
        value=True,
        help="Enable automatic plot generation based on your queries"
    )

    # Smart summary & suggestions toggle
    st.sidebar.header("🧠 Insights")
    smart_summary_enabled = st.sidebar.toggle(
        "Auto Summary & Suggestions",
        value=st.session_state.get('smart_summary_enabled', True),
        help="Generate a short insight and suggest follow-up queries after each run"
    )
    st.session_state.smart_summary_enabled = smart_summary_enabled
    
    # File upload in sidebar
    st.sidebar.header("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file to analyze"
    )
    
    # Show available files
    if os.path.exists('data') and os.listdir('data'):
        st.sidebar.header("📁 Available Files")
        files = os.listdir('data')
        for file in files:
            st.sidebar.text(f"• {file}")
    
    return agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled, uploaded_file

def handle_file_upload(uploaded_file):
    """Handle file upload and show preview dialog"""
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            
            # Save the file
            file_path = f"data/{uploaded_file.name}"
            data.to_csv(file_path, index=False)
            
            st.sidebar.success(f"✅ File uploaded! Shape: {data.shape}")
            
            # Show preview button
            if st.sidebar.button("📋 View Dataset Preview"):
                show_data_preview(data, uploaded_file.name)
            
            # Save data info
            dtypes_dict, summary_stats = save_data_info(data, uploaded_file.name)
            
            # Create SQLite database
            db_path = create_sqlite_db(data, uploaded_file.name)
            
            if st.session_state.debug_mode:
                st.sidebar.success(f"✅ Database created: {db_path}")
            
            # Store in session state
            st.session_state['data'] = data
            st.session_state['db_path'] = db_path
            st.session_state['table_info'] = get_table_info(db_path)
            st.session_state['current_file'] = uploaded_file.name
            
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(traceback.format_exc())
            return False
    return False

def render_plots_section():
    """Render the plots section on the left"""
    if st.session_state.plots_history:
        st.header("📊 Generated Plots")
        
        for i, plot_entry in enumerate(st.session_state.plots_history):
            with st.expander(f"Plot {i+1}: {plot_entry['question'][:50]}...", expanded=(i==0)):
                st.write(f"**Question:** {plot_entry['question']}")
                st.write(f"**Generated:** {plot_entry['timestamp'].strftime('%H:%M:%S')}")
                
                # Display plots
                if 'plot_data' in plot_entry and plot_entry['plot_data']:
                    plots = plot_entry['plot_data']
                    for j, plot_info in enumerate(plots):
                        if 'figure' in plot_info and plot_info['figure']:
                            st.plotly_chart(plot_info['figure'], use_container_width=True, key=f"plot_{i}_{j}")
                            
                            # Show insights if available
                            if 'insights' in plot_info and plot_info['insights']:
                                with st.expander(f"🔍 Insights for Plot {j+1}", expanded=False):
                                    insights = plot_info['insights']
                                    if isinstance(insights, list):
                                        for insight in insights:
                                            st.write(f"• {insight}")
                                    else:
                                        st.write(insights)
                        elif 'config' in plot_info:
                            st.write(f"Plot {j+1}: {plot_info['config'].get('title', 'Untitled')}")
    else:
        st.header("📊 Plots")
        st.info("No plots generated yet. Enable plot generation and ask questions about your data!")

def render_query_panel(agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled):
    """Render the query panel on the right"""
    # Toggle button for hiding/showing query panel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("🔍 Query Your Data")
    with col2:
        if st.button("👁️" if st.session_state.show_query_panel else "👁️‍🗨️"):
            st.session_state.show_query_panel = not st.session_state.show_query_panel
    
    if not st.session_state.show_query_panel:
        return
    
    if 'db_path' not in st.session_state:
        st.info("👆 Please upload a CSV file first to start querying your data.")
        return
    
    # Initialize LLM agent
    try:
        agent = SQLAgent(agent_type, model_name=model, base_url=base_url, api_key=api_key)
        
        # Summary area (above the query input)
        st.subheader("🧠 Summary & Suggestions")
        if smart_summary_enabled and st.session_state.summaries:
            last_summary = st.session_state.summaries[-1]
            st.write(last_summary.get('summary', ''))
            suggestions = last_summary.get('suggestions', [])
            if suggestions:
                st.caption("Suggested next queries:")
                for idx, s in enumerate(suggestions[:3], 1):
                    st.write(f"{idx}. {s}")
        elif smart_summary_enabled:
            st.info("Insights will appear here after you run a query.")
        else:
            st.caption("Auto Summary is off. Turn it on in the sidebar to see insights.")

        # Query input
        user_question = st.text_area(
            "Ask a question about your data:",
            placeholder="e.g., What is the average price by location? Show me the top 10 most expensive houses.",
            height=100,
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            execute_query = st.button("🚀 Generate SQL & Execute", type="primary")
        with col2:
            comprehensive_analysis = st.button("📊 Comprehensive Analysis", type="secondary")
        with col3:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Query History")
            
            for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat_entry['question'][:50]}...", expanded=(i==0)):
                    st.write(f"**Question:** {chat_entry['question']}")
                    st.write(f"**Time:** {chat_entry['timestamp'].strftime('%H:%M:%S')}")
                    
                    if chat_entry['error']:
                        st.error(f"❌ Error: {chat_entry['error']}")
                    else:
                        # Handle multiple queries
                        if isinstance(chat_entry['sql_query'], list):
                            st.write("**Multiple SQL Queries Generated:**")
                            for j, query_info in enumerate(chat_entry['sql_query']):
                                with st.expander(f"Query {j+1}: {query_info.get('description', 'Step ' + str(query_info.get('step', j+1)))}"):
                                    st.code(query_info['query'], language="sql")
                        else:
                            with st.expander("🔧 View SQL Query"):
                                st.code(chat_entry['sql_query'], language="sql")
                        
                        if chat_entry['result'] is not None and not chat_entry['result'].empty:
                            st.write(f"**Results:** {chat_entry['result_shape'][0]} rows × {chat_entry['result_shape'][1]} columns")
                            st.dataframe(chat_entry['result'], use_container_width=True)
                            
                            # Download button for individual query results
                            csv = chat_entry['result'].to_csv(index=False)
                            st.download_button(
                                label="💾 Download",
                                data=csv,
                                file_name=f"query_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key=f"download_{i}"
                            )
                        else:
                            st.info("No results found for this query.")
        
    # Execute comprehensive analysis
        if comprehensive_analysis and user_question:
            with st.spinner("🔬 Performing comprehensive statistical analysis..."):
                try:
                    # Get the current data
                    current_data = st.session_state.get('data')
                    if current_data is None:
                        st.error("❌ No data available for analysis. Please upload a dataset first.")
                        return
                    
                    # Generate comprehensive analysis
                    from src.plotting.plot_generator import generate_comprehensive_analysis, create_comprehensive_visualizations
                    
                    analysis_results = generate_comprehensive_analysis(current_data, user_question)
                    
                    # Create visualizations
                    visualizations = create_comprehensive_visualizations(current_data, analysis_results)
                    
                    # Display analysis results
                    st.subheader("📊 Comprehensive Statistical Analysis")
                    
                    # Summary statistics
                    with st.expander("📈 Summary Statistics", expanded=True):
                        summary_stats = analysis_results.get('summary_stats', {})
                        if 'dataset_info' in summary_stats:
                            info = summary_stats['dataset_info']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Rows", info.get('rows', 0))
                            with col2:
                                st.metric("Columns", info.get('columns', 0))
                            with col3:
                                st.metric("Memory (MB)", f"{info.get('memory_usage', 0) / 1024 / 1024:.2f}")
                            with col4:
                                st.metric("Duplicates", info.get('duplicate_rows', 0))
                    
                    # Data quality analysis
                    with st.expander("🔍 Data Quality Analysis"):
                        quality_issues = analysis_results.get('data_quality', {})
                        if quality_issues.get('missing_values'):
                            st.write("**Missing Values:**")
                            for col, info in quality_issues['missing_values'].items():
                                st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                        
                        if quality_issues.get('outliers'):
                            st.write("**Outliers Detected:**")
                            for col, info in quality_issues['outliers'].items():
                                st.write(f"• {col}: {info['count']} ({info['percentage']:.1f}%)")
                    
                    # Univariate analysis
                    with st.expander("📊 Univariate Analysis"):
                        univariate_results = analysis_results.get('univariate', {})
                        for col, analysis in univariate_results.items():
                            if analysis['type'] == 'numerical':
                                stats = analysis['basic_stats']
                                st.write(f"**{col}** (Numerical)")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Mean", f"{stats['mean']:.2f}")
                                with col2:
                                    st.metric("Median", f"{stats['median']:.2f}")
                                with col3:
                                    st.metric("Std Dev", f"{stats['std']:.2f}")
                                with col4:
                                    st.metric("IQR", f"{stats['iqr']:.2f}")
                            else:
                                stats = analysis['basic_stats']
                                st.write(f"**{col}** (Categorical)")
                                st.write(f"• Unique values: {stats['unique_values']}")
                                st.write(f"• Most common: {stats['most_common']} ({stats['most_common_count']} times)")
                    
                    # Correlation analysis
                    with st.expander("🔗 Correlation Analysis"):
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
                    
                    # Display visualizations
                    st.subheader("📈 Statistical Visualizations")
                    
                    # Create tabs for different types of visualizations
                    if visualizations:
                        tab_names = ["Univariate", "Bivariate", "Multivariate", "Correlations"]
                        tabs = st.tabs(tab_names)
                        
                        univariate_plots = [v for v in visualizations if v['type'] in ['histogram', 'box', 'bar']]
                        bivariate_plots = [v for v in visualizations if v['type'] == 'scatter']
                        multivariate_plots = [v for v in visualizations if v['type'] in ['line', 'heatmap']]
                        correlation_plots = [v for v in visualizations if 'correlation' in v['title'].lower()]
                        
                        with tabs[0]:
                            if univariate_plots:
                                for plot in univariate_plots[:4]:  # Limit to 4 plots
                                    st.plotly_chart(plot['figure'], use_container_width=True)
                                    st.caption(plot['description'])
                            else:
                                st.info("No univariate plots available.")
                        
                        with tabs[1]:
                            if bivariate_plots:
                                for plot in bivariate_plots[:4]:  # Limit to 4 plots
                                    st.plotly_chart(plot['figure'], use_container_width=True)
                                    st.caption(plot['description'])
                            else:
                                st.info("No bivariate plots available.")
                        
                        with tabs[2]:
                            if multivariate_plots:
                                for plot in multivariate_plots[:2]:  # Limit to 2 plots
                                    st.plotly_chart(plot['figure'], use_container_width=True)
                                    st.caption(plot['description'])
                            else:
                                st.info("No multivariate plots available.")
                        
                        with tabs[3]:
                            if correlation_plots:
                                for plot in correlation_plots:
                                    st.plotly_chart(plot['figure'], use_container_width=True)
                                    st.caption(plot['description'])
                            else:
                                st.info("No correlation plots available.")
                    
                    # Add to chat history
                    add_to_chat_history(
                        user_question, 
                        "Comprehensive Analysis", 
                        current_data.head(10),  # Show sample of data
                        timestamp=datetime.now()
                    )
                    
                    # Generate insights
                    if smart_summary_enabled:
                        try:
                            if 'insight_agent' not in st.session_state:
                                st.session_state.insight_agent = InsightAgent(agent)
                            
                            insight_summary = st.session_state.insight_agent.analyze_query_result(
                                user_question, current_data
                            )
                            
                            # Create summary with analysis highlights
                            summary_prompt = f"""
Based on the comprehensive analysis performed, provide:
1. A concise 2-3 sentence summary of key findings
2. 3 specific follow-up analysis suggestions

ANALYSIS RESULTS:
- Dataset: {analysis_results['summary_stats']['dataset_info']['rows']} rows, {analysis_results['summary_stats']['dataset_info']['columns']} columns
- Missing values: {len(analysis_results['data_quality'].get('missing_values', {}))} columns affected
- Strong correlations: {len(analysis_results['correlation_analysis'].get('strong_correlations', []))} found
- Outliers detected: {len(analysis_results['data_quality'].get('outliers', {}))} columns

USER QUESTION: {user_question}

Respond in JSON format:
{{"summary": "your insight", "suggestions": ["suggestion1", "suggestion2", "suggestion3"]}}
"""
                            
                            llm_resp = agent.query(summary_prompt).strip()
                            if llm_resp.startswith('```json'):
                                llm_resp = llm_resp[7:-3]
                            elif llm_resp.startswith('```'):
                                llm_resp = llm_resp[3:-3]
                            
                            try:
                                payload = json.loads(llm_resp)
                                st.session_state.summaries.append({
                                    'timestamp': datetime.now(),
                                    'question': user_question,
                                    'summary': payload.get('summary', ''),
                                    'suggestions': payload.get('suggestions', [])
                                })
                            except:
                                st.session_state.summaries.append({
                                    'timestamp': datetime.now(),
                                    'question': user_question,
                                    'summary': f"Comprehensive analysis completed. Found {len(analysis_results['correlation_analysis'].get('strong_correlations', []))} strong correlations and analyzed {len(analysis_results['univariate'])} variables.",
                                    'suggestions': ["Explore specific correlations", "Analyze outliers", "Investigate missing values"]
                                })
                                
                        except Exception as se:
                            if st.session_state.debug_mode:
                                st.warning(f"Summary generation failed: {se}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during comprehensive analysis: {str(e)}")
                    if st.session_state.debug_mode:
                        st.error(traceback.format_exc())
        
    # Execute new query
        if execute_query and user_question:
            with st.spinner("🤖 Generating SQL queries..."):
                try:
                    # Try to generate multiple SQL queries first
                    multiple_queries = agent.generate_multiple_sql_queries(
                        user_question=user_question,
                        table_info=st.session_state['table_info'],
                        max_queries=5
                    )
                    
                    if len(multiple_queries) > 1:
                        st.success(f"🎯 Generated {len(multiple_queries)} SQL queries for comprehensive analysis!")
                        
                        # Execute all queries
                        all_results = []
                        combined_results = None
                        
                        for i, query_info in enumerate(multiple_queries):
                            sql_query = query_info['query']
                            description = query_info.get('description', f'Query {i+1}')
                            
                            if st.session_state.debug_mode:
                                st.write(f"**Executing Query {i+1}:** {description}")
                                st.code(sql_query, language="sql")
                            
                            # Validate and execute query
                            if sql_query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                                result, error = execute_sql_query(st.session_state['db_path'], sql_query)
                                
                                if error:
                                    st.error(f"❌ Error in Query {i+1}: {error}")
                                elif result is not None and not result.empty:
                                    all_results.append({
                                        'query': query_info,
                                        'result': result,
                                        'description': description
                                    })
                                    
                                    # Combine results for analysis
                                    if combined_results is None:
                                        combined_results = result.copy()
                                    else:
                                        # Try to combine results (this is a simplified approach)
                                        combined_results = pd.concat([combined_results, result], ignore_index=True)
                                    
                                    st.success(f"✅ Query {i+1} executed successfully: {result.shape[0]} rows")
                                else:
                                    st.warning(f"⚠️ Query {i+1} returned no results")
                            else:
                                st.error(f"❌ Query {i+1} is not a valid SELECT/PRAGMA statement")
                        
                        # Add to chat history with multiple queries
                        if all_results:
                            add_to_chat_history(
                                user_question, 
                                multiple_queries, 
                                combined_results if combined_results is not None else all_results[0]['result']
                            )
                            
                            # Generate plots and insights for combined results
                            if enable_plots and combined_results is not None and not combined_results.empty:
                                try:
                                    from src.plotting.plot_generator import generate_plot_suggestions
                                    
                                    columns_info = {col: combined_results[col].dtype for col in combined_results.columns}
                                    plot_suggestions = generate_plot_suggestions(agent, user_question, columns_info, combined_results)
                                    
                                    plots = []
                                    for plot_config in plot_suggestions:
                                        plot_fig = create_plot_figure(plot_config, combined_results)
                                        if plot_fig:
                                            plots.append({
                                                'config': plot_config,
                                                'figure': plot_fig
                                            })
                                    
                                    if plots:
                                        add_plot_to_history(plots, user_question)
                                        
                                except Exception as plot_error:
                                    if st.session_state.debug_mode:
                                        st.error(f"Plot generation error: {plot_error}")
                            
                            # Smart summary for multiple queries
                            if smart_summary_enabled and combined_results is not None:
                                try:
                                    if 'insight_agent' not in st.session_state:
                                        st.session_state.insight_agent = InsightAgent(agent)
                                    
                                    insight_summary = st.session_state.insight_agent.analyze_query_result(
                                        user_question, combined_results
                                    )
                                    
                                    # Create summary for multiple queries
                                    summary_prompt = f"""
Multiple SQL queries were executed to answer: "{user_question}"

QUERIES EXECUTED:
{chr(10).join([f"{i+1}. {q['description']}" for i, q in enumerate(multiple_queries)])}

COMBINED RESULTS:
- Total rows: {combined_results.shape[0]}
- Total columns: {combined_results.shape[1]}
- Analysis: {insight_summary}

Provide a concise summary and 3 follow-up suggestions in JSON format:
{{"summary": "your insight", "suggestions": ["suggestion1", "suggestion2", "suggestion3"]}}
"""
                                    
                                    llm_resp = agent.query(summary_prompt).strip()
                                    if llm_resp.startswith('```json'):
                                        llm_resp = llm_resp[7:-3]
                                    elif llm_resp.startswith('```'):
                                        llm_resp = llm_resp[3:-3]
                                    
                                    try:
                                        payload = json.loads(llm_resp)
                                        st.session_state.summaries.append({
                                            'timestamp': datetime.now(),
                                            'question': user_question,
                                            'summary': payload.get('summary', ''),
                                            'suggestions': payload.get('suggestions', [])
                                        })
                                    except:
                                        st.session_state.summaries.append({
                                            'timestamp': datetime.now(),
                                            'question': user_question,
                                            'summary': f"Executed {len(multiple_queries)} queries successfully. Combined results show {combined_results.shape[0]} rows of data.",
                                            'suggestions': ["Analyze specific query results", "Explore data patterns", "Generate additional visualizations"]
                                        })
                                        
                                except Exception as se:
                                    if st.session_state.debug_mode:
                                        st.warning(f"Summary generation failed: {se}")
                            
                            st.rerun()
                        else:
                            st.error("❌ No successful queries to display.")
                    
                    else:
                        # Fallback to single query (existing logic)
                        sql_query = agent.generate_sql(
                            user_question=user_question,
                            table_info=st.session_state['table_info']
                        )

                        if st.session_state.debug_mode:
                            st.write("**Debug Info:**")
                            st.write(f"Raw query returned: `{sql_query}`")
                        
                        if sql_query:
                            # Validate SQL query before execution
                            if sql_query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                                # Execute the query
                                result, error = execute_sql_query(st.session_state['db_path'], sql_query)
                                
                                if error:
                                    add_to_chat_history(user_question, sql_query, None, error)
                                    st.error(f"❌ SQL Error: {error}")
                                elif result is not None:
                                    # Add to chat history
                                    add_to_chat_history(user_question, sql_query, result)
                                    plot_configs_for_insights = []
                                    # Generate plots if enabled
                                    if enable_plots and not result.empty:
                                        try:
                                            # Generate plot suggestions and create plots
                                            from src.plotting.plot_generator import generate_plot_suggestions
                                            
                                            # Get column information for plot generation
                                            columns_info = {col: result[col].dtype for col in result.columns}
                                            plot_suggestions = generate_plot_suggestions(agent, user_question, columns_info, result)
                                            print(Fore.MAGENTA + json.dumps(plot_suggestions, indent=2) + Fore.RESET)
                                            # Store plot data for later display with insights
                                            plots = []
                                            for plot_config in plot_suggestions:
                                                # Create plot and capture it
                                                plot_fig = create_plot_figure(plot_config, result)
                                                if plot_fig:
                                                    plots.append({
                                                        'config': plot_config,
                                                        'figure': plot_fig
                                                    })
                                                    plot_configs_for_insights.append(plot_config)
                                            
                                            if plots:
                                                # Generate plot insights using InsightAgent
                                                try:
                                                    if 'insight_agent' not in st.session_state:
                                                        st.session_state.insight_agent = InsightAgent(agent)
                                                        
                                                    plot_insights_list = st.session_state.insight_agent.generate_plot_insights(
                                                        plot_configs_for_insights, result, user_question
                                                    )
                                                    print(plot_configs_for_insights)
                                                    print(
                                                        Fore.BLUE + f"Plot insights generated: {plot_insights_list}" + Fore.RESET
                                                    )
                                                    # Add insights to plots - now plot_insights_list is a list of lists
                                                    for i, plot in enumerate(plots):
                                                        if i < len(plot_insights_list) and plot_insights_list[i]:
                                                            plot['insights'] = plot_insights_list[i]
                                                                
                                                except Exception as insight_error:
                                                    if st.session_state.debug_mode:
                                                        st.error(f"Plot insight generation error: {insight_error}")
                                                        st.error(traceback.format_exc())
                                                
                                                add_plot_to_history(plots, user_question)
                                        except Exception as plot_error:
                                            if st.session_state.debug_mode:
                                                st.error(f"Plot generation error: {plot_error}")
                                                st.error(traceback.format_exc())

                                    # Smart summary and suggestions with InsightAgent
                                    if smart_summary_enabled:
                                        try:
                                            # Initialize InsightAgent if not exists
                                            if 'insight_agent' not in st.session_state:
                                                st.session_state.insight_agent = InsightAgent(agent)
                                            plot_insights_list = st.session_state.insight_agent.generate_plot_insights(
                                                        plot_configs_for_insights, result, user_question
                                                    )
                                            print(plot_configs_for_insights)
                                            print(
                                                Fore.BLUE + f"Plot insights generated: {plot_insights_list}" + Fore.RESET
                                            )
                                            # Use InsightAgent for sophisticated analysis
                                            insight_summary = st.session_state.insight_agent.analyze_query_result(
                                                user_question, result
                                            )
                                            print(Fore.CYAN + "Insight Summary: " + insight_summary + Fore.RESET)
                                            # Add to analysis context
                                            st.session_state.analysis_context.append(insight_summary)
                                            context_tail = "\n\n".join(set(st.session_state.analysis_context[-5:]))  # last 5
                                            
                                            summary_prompt = (
                                                "You are a data analyst. Given the latest query results analysis and prior context, "
                                                "write a concise 1-2 sentence insight that includes key statistical findings (means, medians, distributions, correlations). "
                                                "Then suggest up to 3 specific follow-up analysis questions.\n\n"
                                                f"[PRIOR CONTEXT]\n{context_tail}\n\n"
                                                f"[LATEST QUESTION]\n{user_question}\n\n"
                                                f"[LATEST RESULT ANALYSIS]\n{insight_summary}\n\n"
                                                f"[PLOT INSIGHTS]\n{plot_insights_list}\n\n"
                                                "Focus on actionable insights with specific numbers. "
                                                "Respond in JSON with keys: summary (string), suggestions (array of strings)."
                                            )
                                            print(Fore.GREEN + "Summary Prompt: " + summary_prompt + Fore.RESET)
                                            llm_resp = agent.query(summary_prompt).strip()
                                            print("------------------------------------------------")
                                            print(llm_resp)
                                            if llm_resp.startswith('```json'):
                                                llm_resp = llm_resp[7:-3]
                                            elif llm_resp.startswith('```'):
                                                llm_resp = llm_resp[3:-3]
                                            import json as _json
                                            payload = {}
                                            try:
                                                payload = _json.loads(llm_resp)
                                            except Exception:
                                                # fallback: wrap raw text
                                                payload = {"summary": llm_resp[:300], "suggestions": []}
                                            st.session_state.summaries.append({
                                                'timestamp': datetime.now(),
                                                'question': user_question,
                                                'summary': payload.get('summary', ''),
                                                'suggestions': payload.get('suggestions', [])
                                            })
                                        except Exception as se:
                                            if st.session_state.debug_mode:
                                                st.warning(f"Summary generation failed: {se}")
                                    
                                    st.rerun()  # Refresh to show new chat entry
                                else:
                                    add_to_chat_history(user_question, sql_query, None, "Unknown error occurred while executing query")
                                    st.error("❌ Unknown error occurred while executing query.")
                            else:
                                error_msg = "Generated query is not a valid SELECT/PRAGMA statement"
                                add_to_chat_history(user_question, sql_query, None, error_msg)
                                st.error(f"❌ {error_msg}. Please try rephrasing your question.")
                                if st.session_state.debug_mode:
                                    st.write(f"Generated query: `{sql_query}`")
                        else:
                            error_msg = "Could not generate SQL query"
                            add_to_chat_history(user_question, None, None, error_msg)
                            st.error("❌ Could not generate SQL query. Please try rephrasing your question.")
                            
                            # Show some example queries
                            st.info("💡 Try questions like:")
                            st.write("- What are the column names?")
                            st.write("- Show me the first 10 rows")
                            st.write("- What is the average of [column_name]?")
                            st.write("- How many records are there?")
                            st.write("- What are the unique values in [column_name]?")
                            
                except Exception as e:
                    error_msg = str(e)
                    add_to_chat_history(user_question, None, None, error_msg)
                    st.error(f"❌ Error: {error_msg}")
                    if st.session_state.debug_mode:
                        st.error(traceback.format_exc())
        
        elif execute_query and not user_question:
            st.warning("⚠️ Please enter a question about your data.")
    
    except Exception as e:
        st.error(f"❌ Error initializing LLM agent: {str(e)}")
        if st.session_state.debug_mode:
            st.error(traceback.format_exc())

def main():
    st.set_page_config(
        page_title="Auto Data Analysis with LLM",
        page_icon="📊",
        layout="wide"
    )
    os.makedirs("data", exist_ok=True)
    
    # Initialize session state
    initialize_session_state()

    st.title("📊 Auto Data Analysis with LLM")
    
    # Render sidebar
    agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled, uploaded_file = render_sidebar()
    
    # Handle file upload
    handle_file_upload(uploaded_file)
    
    # Main layout: plots on left, queries on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_plots_section()
    
    with col2:
        render_query_panel(agent_type, model, api_key, base_url, enable_plots, smart_summary_enabled)


if __name__ == "__main__":
    main()
