import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from io import BytesIO
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
#from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import mstats
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
import json

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'redo_stack' not in st.session_state:
    st.session_state.redo_stack = []
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Page Config
st.set_page_config(page_title="Advanced Data Analysis Platform", page_icon="📊", layout="wide")

# ================================
# Helper Functions
# ================================

def log_action(action):
    """Log processing actions for audit trail"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_log.append(f"[{timestamp}] {action}")

def update_df(new_df, action=""):
    """Update dataframe with history tracking"""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())
    st.session_state.redo_stack = []
    st.session_state.df = new_df
    if action:
        log_action(action)

@st.cache_data
def load_data(file, nrows = None):
    """Load data with comprehensive error handling"""
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin-1')
            except pd.errors.ParserError:
                file.seek(0)
                df = pd.read_csv(file, sep=';', encoding='utf-8')
        elif file.name.endswith(('.xlsx', 'xls')):
            df = pd.read_excel(file, engine='openpyxl' if file.name.endswith('.xlsx') else 'xlrd')
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        elif file.name.endswith('.parquet'):
            df = pd.read_parquet(file)
        else:
            st.error("Unsupported file format")
            return None
        
        if df.empty or len(df.columns) == 0:
            st.error("Invalid dataset")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_column_types(df):
    """Categorize columns by data type"""
    return {
        'numeric': list(df.select_dtypes(include=[np.number]).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean': list(df.select_dtypes(include=['bool']).columns)
    }

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = np.where(z_scores > threshold)[0]
    return len(outliers)

def generate_column_metadata(df):
    """Generate comprehensive metadata for all columns"""
    metadata = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            type_category = "numeric"
        elif pd.api.types.is_bool_dtype(df[col]) or df[col].nunique() == 2:
            type_category = "binary"
        else:
            type_category = "categorical"
        
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        unique_count = df[col].nunique()
        
        # Determine relevance
        target_keywords = ['target', 'label', 'class', 'status', 'outcome', 'result']
        if any(k in col.lower() for k in target_keywords):
            relevance = "Target"
        elif missing_pct > 50:
            relevance = "Low"
        elif type_category in ['numeric', 'binary'] and missing_pct <= 30:
            relevance = "High"
        elif type_category == 'categorical' and unique_count <= 20:
            relevance = "High"
        else:
            relevance = "Medium"
        
        metadata.append({
            'Column': col,
            'Type': type_category,
            'Data Type': col_type,
            'Missing %': f"{missing_pct:.1f}%",
            'Unique Values': unique_count,
            'Relevance': relevance
        })
    
    return pd.DataFrame(metadata)

def calculate_data_quality_score(df):
    """Calculate overall data quality score based on 6 dimensions"""
    scores = {}
    
    # 1. Completeness (20%)
    completeness = 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    scores['Completeness'] = completeness
    
    # 2. Validity (20%) - based on data type consistency
    validity = 100  # Simplified - assume loaded data is valid
    scores['Validity'] = validity
    
    # 3. Uniqueness (10%) - based on duplicate rate
    uniqueness = 100 - (df.duplicated().sum() / len(df) * 100)
    scores['Uniqueness'] = uniqueness
    
    # 4. Consistency (15%) - simplified check
    consistency = 90  # Placeholder
    scores['Consistency'] = consistency
    
    # 5. Accuracy (25%) - placeholder
    accuracy = 85  # Would require validation against source
    scores['Accuracy'] = accuracy
    
    # 6. Timeliness (10%) - placeholder
    timeliness = 95  # Would require checking data freshness
    scores['Timeliness'] = timeliness
    
    # Weighted average
    weights = {'Completeness': 0.20, 'Validity': 0.20, 'Accuracy': 0.25, 
               'Consistency': 0.15, 'Timeliness': 0.10, 'Uniqueness': 0.10}
    
    overall_score = sum(scores[k] * weights[k] for k in scores.keys())
    
    return overall_score, scores

def get_columns(dataset_name):
    datasets = st.session_state.get("datasets", {})
    if dataset_name in datasets:
        return list(datasets[dataset_name]["df"].columns)
    return []


# ---------- Helper function to validate relationships ----------
def check_valid_relationship(df_left, col_left, df_right, col_right):
    # Check columns exist
    if col_left not in df_left.columns:
        return False, f"Column '{col_left}' not found in left dataset"
    if col_right not in df_right.columns:
        return False, f"Column '{col_right}' not found in right dataset"

    # Check data types compatible (simple check)
    if df_left[col_left].dtype != df_right[col_right].dtype:
        return False, "Data types of the columns do not match"

    # Check some overlap in values
    left_values = set(df_left[col_left].dropna().unique())
    right_values = set(df_right[col_right].dropna().unique())
    overlap = left_values.intersection(right_values)
    if not overlap:
        return False, "No overlapping values between the columns"

    # Optional: check uniqueness for "one" side relationships
    # Here we allow the user to specify relationship type, so we skip strict uniqueness check
    return True, "Valid relationship"

# ----------- Visualization function ------------
def visualize_relationships(relationships):
    if not relationships:
        st.info("No relationships to visualize.")
        return

    G = nx.Graph()

    # Add datasets as nodes
    datasets = set()
    for rel in relationships:
        datasets.add(rel["Left Dataset"])
        datasets.add(rel["Right Dataset"])

    for ds in datasets:
        G.add_node(ds, color='skyblue', size=1500)

    # Add edges with labels
    for rel in relationships:
        left = rel["Left Dataset"]
        right = rel["Right Dataset"]
        rel_type = rel["Relationship"]
        G.add_edge(left, right, label=rel_type)

    pos = nx.spring_layout(G, seed=42)
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels (relationship types)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.axis('off')
    st.pyplot(plt)
    plt.close()

# ------------ The Data Modeling tab content ------------
def data_modeling_tab():
    st.header("🔗 Data Modeling")

    datasets = st.session_state.get("datasets", {})

    if len(datasets) < 2:
        st.warning("Upload at least two datasets to create relationships.")
        return

    if "relationships" not in st.session_state:
        st.session_state["relationships"] = []

    # Select datasets to link
    selected_left = st.selectbox("Select Left Dataset", list(datasets.keys()))
    selected_right = st.selectbox("Select Right Dataset", list(datasets.keys()), index=1)

    # Prevent linking dataset to itself
    if selected_left == selected_right:
        st.error("Please select two different datasets.")
        return

    left_df = datasets[selected_left]["df"]
    right_df = datasets[selected_right]["df"]

    # Select columns from datasets
    left_col = st.selectbox(f"Select column from '{selected_left}'", left_df.columns)
    right_col = st.selectbox(f"Select column from '{selected_right}'", right_df.columns)

    # Relationship type
    rel_type = st.selectbox("Relationship Type", ["One-to-One", "One-to-Many", "Many-to-One", "Many-to-Many"])

    if st.button("➕ Add Relationship"):
        valid, msg = check_valid_relationship(left_df, left_col, right_df, right_col)
        if not valid:
            st.error(f"Cannot add relationship: {msg}")
        else:
            new_rel = {
                "Left Dataset": selected_left,
                "Left Column": left_col,
                "Right Dataset": selected_right,
                "Right Column": right_col,
                "Relationship": rel_type
            }
            if new_rel not in st.session_state["relationships"]:
                st.session_state["relationships"].append(new_rel)
                st.success("Relationship added!")
            else:
                st.info("This relationship already exists.")

    # Show defined relationships
    if st.session_state["relationships"]:
        st.subheader("Defined Relationships")
        rel_df = pd.DataFrame(st.session_state["relationships"])
        st.dataframe(rel_df, use_container_width=True)

    else:
        st.info("No relationships defined yet.")

# Initialize session state for popups
if "show_python_editor" not in st.session_state:
    st.session_state.show_python_editor = False
if "show_sql_editor" not in st.session_state:
    st.session_state.show_sql_editor = False
if "python_preview" not in st.session_state:
    st.session_state.python_preview = None
if "sql_preview" not in st.session_state:
    st.session_state.sql_preview = None

# ---------------- Sidebar buttons ----------------
with st.sidebar:
    st.header("Code Editors")
    if st.button("Open Python Editor"):
        st.session_state.show_python_editor = True
    if st.button("Open SQL Editor"):
        st.session_state.show_sql_editor = True



# ================================
# MAIN APP
# ================================

st.title("📊 Data Analysis")
st.markdown("**Comprehensive data analysis, cleaning, and transformation toolkit**")

# Sidebar - Navigation
with st.sidebar:
    st.header("📑 Navigation")
    page = st.radio("Go to", [
        "🏠 Dataset",
        "🔄 Transformations",
        "🔍 Data Profiling",
        "🧹 Data Cleaning",
        "📈 Feature Engineering",
        "📊 Visualizations",
        "💾 Export & Reports"
    ])
    
    st.markdown("---")
    
    # Undo/Redo
    if st.button("↩️ Undo"):
        if st.session_state.history:
            st.session_state.redo_stack.append(st.session_state.df.copy())
            st.session_state.df = st.session_state.history.pop()
            st.rerun()
    
    if st.button("↪️ Redo"):
        if st.session_state.redo_stack:
            st.session_state.history.append(st.session_state.df.copy())
            st.session_state.df = st.session_state.redo_stack.pop()
            st.rerun()
    
    st.markdown("---")
    st.caption(f"History: {len(st.session_state.history)} actions")
    
    st.markdown("---")
    with st.expander("📋 View Dataset", expanded=True):
        df = st.session_state.get('df', None)
        if df is not None:
            # Sorting options
            sort_col = st.selectbox("Order by column", options=df.columns, index=0)
            sort_order = st.radio("Sort order", ["Ascending", "Descending"], index=0)
            n_rows = st.slider("Number of rows to display", min_value=5, max_value=100, value=10)

            # Sort dataframe accordingly
            ascending = True if sort_order == "Ascending" else False
            sorted_df = df.sort_values(by=sort_col, ascending=ascending)

            # Show top rows after sort
            st.dataframe(sorted_df.head(n_rows), use_container_width=True)
        else:
            st.write("No dataset loaded yet.")

# ================================
# PAGE 1: HOME & UPLOAD
# ================================
if page == "🏠 Dataset":
    st.header("1. 📁 Upload Your Dataset")

    # --------------------------------
    # Initialize session state
    # --------------------------------
    for key in [
        "df",
        "original_df",
        "current_file",
        "preview_df",
        "preview_file",
        "import_columns",
        "uploaded_file",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # --------------------------------
    # File uploader
    # --------------------------------
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="Supported formats: CSV, Excel, JSON, Parquet",
    )

    # --------------------------------
    # Preview dataset (FIRST 10 ROWS)
    # --------------------------------
    if uploaded_file is not None:

        # Store file safely for reruns
        st.session_state.uploaded_file = uploaded_file

        # Only preview if new file
        if st.session_state.preview_file != uploaded_file.name:

            with st.spinner("Previewing dataset..."):
                preview_df = load_data(uploaded_file, nrows=10)

            if preview_df is None:
                st.error("❌ Failed to preview dataset")
                st.stop()

            st.session_state.preview_df = preview_df
            st.session_state.preview_file = uploaded_file.name
            st.session_state.import_columns = list(preview_df.columns)

    # --------------------------------
    # Preview UI
    # --------------------------------
    if st.session_state.preview_df is not None:

        st.subheader("👀 Preview Before Import")
        st.caption("Data Preview. Select columns to import.")

        st.dataframe(
            st.session_state.preview_df,
            use_container_width=True
        )

        # --------------------------------
        # Column selection
        # --------------------------------
        st.subheader("📋 Select Columns to Import")

        search_term = st.text_input(
            "Search columns",
            placeholder="Type column name..."
        )

        all_columns = list(st.session_state.preview_df.columns)

        if search_term:
            filtered_columns = [
                col for col in all_columns
                if search_term.lower() in col.lower()
            ]
        else:
            filtered_columns = all_columns

        # Streamlit-safe defaults
        valid_defaults = [
            col for col in (st.session_state.import_columns or [])
            if col in filtered_columns
        ]

        selected_columns = st.multiselect(
            "Columns",
            options=filtered_columns,
            default=valid_defaults
        )

        st.session_state.import_columns = selected_columns

        # --------------------------------
        # Import button
        # --------------------------------
        if st.button("✅ Import Selected Columns", type="primary"):

            if not st.session_state.import_columns:
                st.error("Please select at least one column")
                st.stop()

            with st.spinner("Loading selected columns..."):
                full_df = load_data(st.session_state.uploaded_file)

            if full_df is None:
                st.error("❌ Failed to load dataset")
                st.stop()

            final_df = full_df[st.session_state.import_columns]

            update_df(
                final_df,
                f"Imported {len(final_df.columns)} columns from {st.session_state.preview_file}"
            )

            st.session_state.df = final_df
            st.session_state.original_df = final_df.copy()
            st.session_state.current_file = st.session_state.preview_file

            # Cleanup preview state
            for key in [
                "preview_df",
                "preview_file",
                "import_columns",
                "uploaded_file",
            ]:
                st.session_state[key] = None

            st.success("✅ Dataset imported successfully")
            st.rerun()

    # --------------------------------
    # Dataset overview (AFTER IMPORT)
    # --------------------------------
    df = st.session_state.df

    if df is not None:

        st.header("3. 📊 Dataset Overview")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
            st.metric("Memory", f"{memory_mb:.2f} MB")
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        with col5:
            missing_pct = (
                df.isnull().sum().sum()
                / (len(df) * len(df.columns))
            ) * 100
            st.metric("Missing", f"{missing_pct:.1f}%")

        st.subheader("👀 Data Preview")

        rows = st.slider(
            "Rows to display",
            5,
            min(100, len(df)),
            10
        )

        view = st.selectbox(
            "View",
            ["Head", "Tail", "Random"]
        )

        if view == "Head":
            st.dataframe(df.head(rows), use_container_width=True)
        elif view == "Tail":
            st.dataframe(df.tail(rows), use_container_width=True)
        else:
            st.dataframe(
                df.sample(min(rows, len(df))),
                use_container_width=True
            )

# ================================
# PAGE 4: TRANSFORMATIONS
# ================================
elif page == "🔄 Transformations":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Data Transformations")
    
    tab1, tab2 = st.tabs([
        "🔄 Reshaping",
        "✏️ Column Operations"
    ])
    
    import numpy as np
    from scipy.stats import skew
    
    with tab1:
        st.subheader("Data Reshaping")
        with st.popover("ℹ️ What is Data Reshaping?"):
                st.markdown(
                    """
                    Data reshaping changes the structure of your dataset without changing the actual data.
                    It helps reorganize data into formats that are easier to analyze, summarize, or visualize.

                    **Reshaping Operations:**
                    - **Pivot:** Converts data from a long format into a wide format by summarizing values.  
                      *For example, turning rows of sales data into a table with categories as columns.*

                    - **Melt (Unpivot):** Converts data from a wide format into a long format by stacking columns into rows.  
                      *Useful when preparing data for visualization or machine learning.*

                    Pivoting is commonly used for reporting and aggregation, while melting is useful when models or charts expect a long, tidy format.

                    👉 Data reshaping improves flexibility, enables better analysis, and ensures compatibility with different analytical and modeling workflows.
                    """
                )
        
        reshape_type = st.radio("Select Operation", ["Pivot", "Melt (Unpivot)"])
        
        if reshape_type == "Pivot":
            st.write("**Create Pivot Table**")
            
            pivot_col1, pivot_col2, pivot_col3 = st.columns(3)
            
            with pivot_col1:
                index_cols = st.multiselect("Index (rows)", df.columns, key="pivot_index")
            with pivot_col2:
                columns_col = st.selectbox("Columns (pivot)", df.columns, key="pivot_columns")
            with pivot_col3:
                values_col = st.selectbox("Values", df.columns, key="pivot_values")
                aggfunc = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"])
            
            if st.button("🔄 Create Pivot", type="primary"):
                try:
                    pivot_df = df.pivot_table(
                        index=index_cols,
                        columns=columns_col,
                        values=values_col,
                        aggfunc=aggfunc,
                        fill_value=0
                    ).reset_index()
                    
                    update_df(pivot_df, "Created pivot table")
                    st.success(f"✅ Pivot table created! Shape: {pivot_df.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        else:  # Melt
            st.write("**Melt Data (Long Format)**")
            
            melt_col1, melt_col2 = st.columns(2)
            
            with melt_col1:
                id_vars = st.multiselect("ID Variables (keep)", df.columns, key="melt_id")
                value_vars = st.multiselect("Value Variables (unpivot)", df.columns, key="melt_value")
            
            with melt_col2:
                var_name = st.text_input("Variable Column Name", value="variable")
                value_name = st.text_input("Value Column Name", value="value")
            
            if st.button("🔄 Melt Data", type="primary"):
                try:
                    if not value_vars:
                        value_vars = [col for col in df.columns if col not in id_vars]
                    
                    melt_df = pd.melt(
                        df,
                        id_vars=id_vars if id_vars else None,
                        value_vars=value_vars,
                        var_name=var_name,
                        value_name=value_name
                    )
                    
                    update_df(melt_df, "Melted data")
                    st.success(f"✅ Data melted! Shape: {melt_df.shape}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Column Operations")
        with st.popover("ℹ️ What are Column Operations?"):
                st.markdown(
                    """
                    Column operations allow you to manage and modify the structure and meaning of individual columns in your dataset.

                    **Available Operations:**
                    - **Rename Columns:** Change column names to be clearer, shorter, or more meaningful.
                      *Helps improve readability and consistency.*

                    - **Drop Columns:** Permanently remove one or more columns from the dataset.
                      *Useful for removing irrelevant or sensitive data.*

                    - **Reorder Columns:** Change the order of columns to improve layout or logical flow.
                      *Helpful for reporting and review.*

                    - **Change Data Type:** Convert a column to a different data type (e.g., number, text, date).
                      *Ensures correct calculations and prevents analysis errors.*

                    - **Create Calculated Column:** Create a new column using a formula based on existing columns.
                      *Enables feature creation and custom metrics.*

                    👉 Proper column management keeps your dataset clean, consistent, and easier to analyze or model.
                    """
                )
        
        operation = st.selectbox("Select Operation", [
            "Rename Columns",
            "Drop Columns",
            "Reorder Columns",
            "Change Data Type",
            "Create Calculated Column"
        ])
        
        if operation == "Rename Columns":
            old_name = st.selectbox("Select column", df.columns)
            new_name = st.text_input("New name")
            
            if new_name and st.button("✏️ Rename", type="primary"):
                df = df.rename(columns={old_name: new_name})
                update_df(df, f"Renamed {old_name} to {new_name}")
                st.success(f"✅ Renamed successfully")
                st.rerun()
        
        elif operation == "Drop Columns":
            cols_to_drop = st.multiselect("Select columns to drop", df.columns)
            
            if cols_to_drop and st.button("🗑️ Drop", type="primary"):
                df = df.drop(columns=cols_to_drop)
                update_df(df, f"Dropped {len(cols_to_drop)} columns")
                st.success(f"✅ Dropped {len(cols_to_drop)} columns")
                st.rerun()
        
        elif operation == "Change Data Type":
            target_col = st.selectbox("Select column", df.columns)
            new_dtype = st.selectbox("New data type", [
                "int", "float", "str", "datetime", "category", "bool"
            ])
            
            if st.button("🔄 Convert", type="primary"):
                try:
                    if new_dtype == "datetime":
                        df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
                    elif new_dtype == "int":
                        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').astype('Int64')
                    elif new_dtype == "float":
                        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                    elif new_dtype == "category":
                        df[target_col] = df[target_col].astype('category')
                    elif new_dtype == "bool":
                        df[target_col] = df[target_col].astype(bool)
                    else:
                        df[target_col] = df[target_col].astype(str)
                    
                    update_df(df, f"Converted {target_col} to {new_dtype}")
                    st.success(f"✅ Data type changed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif operation == "Create Calculated Column":
                new_col_name = st.text_input("New column name", key="calc_col_name")
                expression = st.text_area("Expression (use column names)",
                                         placeholder="e.g., col1 + col2 * 100", key="calc_expression")
                
                if new_col_name and expression and st.button("➕ Create", type="primary"):
                        try:
                                df[new_col_name] = df.eval(expression)
                                update_df(df, f"Created calculated column: {new_col_name}")
                                st.success(f"✅ Column '{new_col_name}' created")
                                st.rerun()
                        except Exception as e:
                                st.error(f"Error: {str(e)}")


# ================================
# PAGE 2: DATA PROFILING
# ================================
elif page == "🔍 Data Profiling":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Data Profiling & Quality Assessment")

    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Statistical Summary",
        "🔍 Column Analysis",
        "📉 Distribution Analysis",
        "🔗 Correlation Analysis"
    ])
    
    with tab1:
        st.subheader("Statistical Summary")
        with st.popover("ℹ️ What is Statistical Summary?"):
                st.markdown(
                    """
                    The **Statistical Summary** provides a high-level quantitative overview of your dataset
                    by separating **numeric** and **categorical** variables.

                    **For numeric columns**, this section:
                    - Computes core descriptive statistics (mean, std, min, max, quartiles)
                    - Measures **skewness** to detect asymmetry in data distribution. 
                      *Skewness tells you if the data leans more to the left or right rather than being perfectly balanced.*
                    - Measures **kurtosis** to identify heavy-tailed or peaked distributions. 
                      *Kurtosis indicates whether your data has more extreme values (outliers) than a normal distribution.*

                    **For categorical columns**, this section:
                    - Counts unique values to assess cardinality
                    - Identifies the most frequent category and its occurrence
                    - Highlights missing values and their proportion

                    👉 Use this summary to quickly detect data quality issues,
                    distribution anomalies, and columns that may require cleaning
                    or transformation before modeling.
                    """
                )

        
        col_types = get_column_types(df)
        
        if col_types['numeric']:
            st.write("**Numeric Columns:**")
            numeric_stats = df[col_types['numeric']].describe().T
            numeric_stats['skewness'] = df[col_types['numeric']].skew()
            numeric_stats['kurtosis'] = df[col_types['numeric']].kurtosis()
            
            st.dataframe(numeric_stats, use_container_width=True)
        
        if col_types['categorical']:
            st.write("**Categorical Columns:**")
            cat_info = []
            for col in col_types['categorical']:
                cat_info.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Most Common Count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                })
            
            st.dataframe(pd.DataFrame(cat_info), use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Detailed Column Analysis")
        with st.popover("ℹ️ What is Detailed Column Analysis?"):
                st.markdown(
                    """
                    This section provides an in-depth look at a selected column from your dataset.

                    **For numeric columns**, the **Statistical Measures** shown are:
                    - Mean: The average value
                    - Median: The middle value when sorted
                    - Standard Deviation: How spread out the values are
                    - Minimum and Maximum values
                    - Skewness: Shows if data leans more to one side (left or right). 
                      *Positive skew means tail on the right; negative means tail on the left.*
                    - Kurtosis: Indicates if the data has more extreme values (outliers) than usual. 
                      *High kurtosis means more outliers.*

                    **For non-numeric columns**, it shows the top 10 most frequent values and their counts.

                    👉 Use this analysis to understand each column’s characteristics, identify issues, 
                    and guide cleaning or feature engineering steps.
                    """
                )
        
        selected_col = st.selectbox("Select column for detailed analysis", df.columns)
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"• Data Type: {df[selected_col].dtype}")
                st.write(f"• Non-Null Count: {df[selected_col].count():,}")
                st.write(f"• Null Count: {df[selected_col].isnull().sum():,}")
                st.write(f"• Unique Values: {df[selected_col].nunique():,}")
            
            with col2:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    st.write("**Statistical Measures:**")
                    st.write(f"• Mean: {df[selected_col].mean():.2f}")
                    st.write(f"• Median: {df[selected_col].median():.2f}")
                    st.write(f"• Std Dev: {df[selected_col].std():.2f}")
                    st.write(f"• Min: {df[selected_col].min():.2f}")
                    st.write(f"• Max: {df[selected_col].max():.2f}")
                    st.write(f"• Skewness: {df[selected_col].skew():.2f}")
                    st.write(f"• Kurtosis: {df[selected_col].kurtosis():.2f}")
                else:
                    st.write("**Value Counts (Top 10):**")
                    value_counts = df[selected_col].value_counts().head(10)
                    st.dataframe(value_counts, use_container_width=True)
    
    with tab3:
        st.subheader("Distribution Analysis")
        with st.popover("ℹ️ What is Distribution Analysis?"):
                st.markdown(
                    """
                    Distribution Analysis helps you understand how data in a numeric column is distributed using four key plots:

                    - **Histogram:** Shows the frequency of data points within value ranges (bins). 
                      *Great for spotting skewness, modality, and gaps.*

                    - **Box Plot:** Visualizes the spread, central tendency, and potential outliers.
                      *The box shows the middle 50% of data; whiskers extend to range; dots indicate outliers.*

                    - **Q-Q Plot:** Compares your data’s distribution against a normal distribution.
                      *Points close to the diagonal line suggest normality; deviations indicate non-normality.*

                    - **Density Plot (KDE):** Smooth estimate of the data’s distribution curve.
                      *Helps visualize the shape of the distribution without binning.*

                    👉 Use these plots to identify skewness, outliers, normality, and overall shape of your data,
                    guiding decisions on transformations or modeling.
                    """
                )

        
        numeric_cols = col_types['numeric']
        if not numeric_cols:
            st.info("No numeric columns available")
        else:
            dist_col = st.selectbox("Select column", numeric_cols, key="dist_col")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogram
            axes[0, 0].hist(df[dist_col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title(f'Histogram of {dist_col}')
            axes[0, 0].set_xlabel(dist_col)
            axes[0, 0].set_ylabel('Frequency')
            
            # Box plot
            axes[0, 1].boxplot(df[dist_col].dropna())
            axes[0, 1].set_title(f'Box Plot of {dist_col}')
            axes[0, 1].set_ylabel(dist_col)
            
            # Q-Q plot
            stats.probplot(df[dist_col].dropna(), dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title(f'Q-Q Plot of {dist_col}')
            
            # KDE plot
            df[dist_col].dropna().plot(kind='density', ax=axes[1, 1])
            axes[1, 1].set_title(f'Density Plot of {dist_col}')
            axes[1, 1].set_xlabel(dist_col)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with tab4:
        st.subheader("Correlation Analysis")
        with st.popover("ℹ️ What is Correlation Analysis?"):
                st.markdown(
                    """
                    Correlation Analysis measures how strongly pairs of numeric variables move together.

                    - **Pearson correlation** measures linear relationships.
                    - **Spearman correlation** measures monotonic relationships (any consistent trend).
                    - **Kendall correlation** measures ordinal associations.

                    The heatmap visually displays correlations between variables:
                    - Colors near **red** or **blue** show strong positive or negative relationships.
                    - Values near **0** indicate weak or no relationship.

                    The table lists variable pairs with **high correlation (|r| > 0.7)**:
                    - Helps identify redundant features or strong predictors.
                    - Useful for feature selection and avoiding multicollinearity in models.

                    👉 Use this analysis to understand variable dependencies and guide modeling decisions.
                    """
                )
        
        numeric_cols = col_types['numeric']
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis")
        else:
            corr_method = st.selectbox("Correlation Method", ["Pearson", "Spearman", "Kendall"])
            
            if corr_method == "Pearson":
                corr_matrix = df[numeric_cols].corr(method='pearson')
            elif corr_method == "Spearman":
                corr_matrix = df[numeric_cols].corr(method='spearman')
            else:
                corr_matrix = df[numeric_cols].corr(method='kendall')
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, fmt='.2f', ax=ax)
            ax.set_title(f'{corr_method} Correlation Matrix')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # High correlations
            st.subheader("🔍 High Correlations (|r| > 0.7)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True, hide_index=True)
            else:
                st.info("No high correlations found")

# ================================
# PAGE 3: DATA CLEANING
# ================================
elif page == "🧹 Data Cleaning":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Data Cleaning Operations")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Missing Values",
        "👥 Duplicates",
        "📊 Outliers",
        "🧹 General Cleaning"
    ])
    
    with tab1:
        st.subheader("Missing Value Treatment")

        with st.popover("ℹ️ What is Missing Value Treatment?"):
            st.markdown(
                """
                Missing values are gaps where data is absent in your dataset.
                This section helps you understand and apply the most suitable
                treatment based on data type and context.
                """
            )

        # ---------------------------------
        # Missing value summary
        # ---------------------------------
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()

        if not missing_cols:
            st.success("✅ No missing values found")

        else:
            missing_df = pd.DataFrame({
                "Column": missing_cols,
                "Missing Count": [missing_summary[col] for col in missing_cols],
                "Missing %": [(missing_summary[col] / len(df) * 100) for col in missing_cols]
            }).sort_values("Missing %", ascending=False)

            st.dataframe(missing_df, use_container_width=True, hide_index=True)

            # ---------------------------------
            # Apply treatment
            # ---------------------------------
            st.subheader("Apply Treatment")

            col1, col2 = st.columns(2)

            with col1:
                target_col = st.selectbox("Select column", missing_cols)

            is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

            with col2:
                if is_numeric:
                    st.info("📊 Recommended: **Fill with Median**")
                else:
                    st.info("🏷️ Recommended: **Fill with Mode** or **Constant**")

            # ---------------------------------
            # Categorised treatment options
            # ---------------------------------
            category = st.selectbox(
                "Treatment Category",
                [
                    "🧹 Remove Data",
                    "📊 Simple Fill (Recommended)",
                    "🔁 Sequential / Time-based",
                    "🧠 Advanced (Context-aware)"
                ]
            )

            ACTIONS = {
                "🧹 Remove Data": [
                    "Drop Rows",
                    "Drop Column"
                ],
                "📊 Simple Fill (Recommended)": [
                    "Fill with Median",
                    "Fill with Mean",
                    "Fill with Mode",
                    "Fill with Constant"
                ],
                "🔁 Sequential / Time-based": [
                    "Forward Fill",
                    "Backward Fill",
                    "Interpolate"
                ],
                "🧠 Advanced (Context-aware)": [
                    "Fill with Grouped Mean",
                    "Fill with Grouped Median",
                    "Fill with Grouped Mode"
                ]
            }

            action = st.selectbox("Treatment Method", ACTIONS[category])

            # ---------------------------------
            # Extra inputs
            # ---------------------------------
            fill_value = None
            group_col = None

            if action == "Fill with Constant":
                if is_numeric:
                    fill_value = st.number_input("Fill value")
                else:
                    fill_value = st.text_input("Fill value", value="Unknown")

            if action.startswith("Fill with Grouped"):
                possible_group_cols = [c for c in df.columns if c != target_col]
                group_col = st.selectbox("Group by column", possible_group_cols)
                st.warning("⚠️ Use only if grouping logically explains missing values")

            # ---------------------------------
            # Apply action
            # ---------------------------------
            if st.button("✅ Apply Treatment", type="primary"):
                try:
                    original_shape = df.shape

                    if action == "Drop Rows":
                        df = df.dropna(subset=[target_col])

                    elif action == "Drop Column":
                        df = df.drop(columns=[target_col])

                    elif action == "Fill with Mean":
                        df[target_col] = df[target_col].fillna(df[target_col].mean())

                    elif action == "Fill with Median":
                        df[target_col] = df[target_col].fillna(df[target_col].median())

                    elif action == "Fill with Mode":
                        df[target_col] = df[target_col].fillna(df[target_col].mode()[0])

                    elif action == "Forward Fill":
                        df[target_col] = df[target_col].ffill()

                    elif action == "Backward Fill":
                        df[target_col] = df[target_col].bfill()

                    elif action == "Interpolate":
                        df[target_col] = df[target_col].interpolate()

                    elif action == "Fill with Constant":
                        df[target_col] = df[target_col].fillna(fill_value)

                    elif action == "Fill with Grouped Mean":
                        df[target_col] = df[target_col].fillna(
                            df.groupby(group_col)[target_col].transform("mean")
                        )

                    elif action == "Fill with Grouped Median":
                        df[target_col] = df[target_col].fillna(
                            df.groupby(group_col)[target_col].transform("median")
                        )

                    elif action == "Fill with Grouped Mode":
                        def mode_func(x):
                            m = x.mode()
                            return m[0] if not m.empty else np.nan

                        df[target_col] = df[target_col].fillna(
                            df.groupby(group_col)[target_col].transform(mode_func)
                        )

                    update_df(df, f"{action} applied on {target_col}")
                    st.success(f"✅ Done! Shape: {original_shape} → {df.shape}")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Duplicate Record Treatment")
        with st.popover("ℹ️ What is Duplicate Record Treatment?"):
                st.markdown(
                    """
                    Duplicate records are repeated rows in your dataset that can bias analysis and modeling. 
                    This section helps you identify and handle these duplicates effectively.

                    **Treatment Methods:**
                    - **Keep First:** Keeps the first occurrence of each duplicate and removes the rest
                    - **Remove All:** Removes all duplicates, leaving unique rows.
                    - **Mark as Duplicate:** Adds a new column marking which rows are duplicates without removing them.

                    👉 Handling duplicates ensures your data is clean and reduces errors in your analysis or model results.
                    """
                )

        
        duplicates = df.duplicated().sum()
        
        if duplicates == 0:
            st.success("✅ No duplicates found")
        else:
            st.warning(f"⚠️ Found {duplicates:,} duplicates ({duplicates/len(df)*100:.1f}%)")
            
            dup_col1, dup_col2 = st.columns(2)
            
            with dup_col1:
                if st.button("👀 View Duplicates"):
                    dup_records = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
                    st.dataframe(dup_records, use_container_width=True)
            
            with dup_col2:
                dup_method = st.selectbox("Method", [
                    "Keep First",
                    "Remove All",
                    "Mark as Duplicate"
                ])
                
                if st.button("🧹 Handle Duplicates", type="primary"):
                    original_len = len(df)
                    
                    if dup_method == "Keep First":
                        df = df.drop_duplicates(keep='first')
                    elif dup_method == "Remove All":
                        df = df.drop_duplicates(keep=False)
                    else:
                        df['is_duplicate'] = df.duplicated(keep=False)
                    
                    update_df(df, f"Handled duplicates: {dup_method}")
                    st.success(f"✅ Removed {original_len - len(df):,} records")
                    st.rerun()
    
    with tab3:
        st.subheader("Outlier Detection & Treatment")
        with st.popover("ℹ️ What is Outlier Detection & Treatment?"):
            st.markdown(
                """
                Outliers are data points that differ significantly from other observations and can distort analysis or model performance. This section helps you detect and handle outliers in numeric columns.

                **Detection Methods:**
                - **Statistical:** IQR (Interquartile Range), Z-Score, Modified Z-Score
                - **Custom:** Specify your own lower and upper bounds

                **Treatment Options:**
                - **Removal:** Remove outliers from dataset
                - **Capping:** Cap outliers to boundary values
                - **Winsorizing:** Limit extreme values to reduce impact without removing data
                - **Transformation:** Logarithmic transform to reduce skewness

                👉 Recommended: Use **IQR** detection combined with **Cap to Bounds** or **Winsorize** treatment for most cases.
                """
            )

        col_types = get_column_types(df)
        numeric_cols = col_types['numeric']

        if not numeric_cols:
            st.info("No numeric columns available")
        else:
            outlier_col1, outlier_col2 = st.columns([2, 1])

            with outlier_col1:
                target_col = st.selectbox("Select numeric column", numeric_cols, key="outlier_col")

                col_min = df[target_col].min()
                col_max = df[target_col].max()
                st.markdown(f"**Column Range:** min = `{col_min:.4f}`, max = `{col_max:.4f}`")

                st.markdown("### Detection Method Category")
                detection_category = st.selectbox(
                    "Choose detection method category",
                    ["Statistical Methods", "Custom Bounds"]
                )

                if detection_category == "Statistical Methods":
                    method = st.selectbox(
                        "Select detection method",
                        ["IQR (Recommended)", "Z-Score", "Modified Z-Score"]
                    )
                else:
                    method = "Custom Bounds"
                    lower_bound = st.number_input("Enter Lower Bound", value=float(col_min))
                    upper_bound = st.number_input("Enter Upper Bound", value=float(col_max))

            with outlier_col2:
                st.markdown("### Treatment Category")
                treatment_category = st.selectbox(
                    "Choose treatment category",
                    ["Removal", "Capping", "Winsorizing", "Transformation"]
                )

                if treatment_category == "Removal":
                    treatment = "Remove"
                elif treatment_category == "Capping":
                    treatment = "Cap to Bounds"
                elif treatment_category == "Winsorizing":
                    treatment = "Winsorize"
                else:
                    treatment = "Log Transform"

            if target_col:
                if method == "IQR (Recommended)":
                    outlier_info = detect_outliers_iqr(df, target_col)
                    if outlier_info:
                        count, lower, upper = outlier_info
                        st.write(f"**Outliers Found:** {count:,} ({count/len(df)*100:.2f}%)")
                        st.write(f"**Bounds:** [{lower:.4f}, {upper:.4f}]")

                elif method == "Z-Score":
                    threshold = st.slider("Z-Score Threshold", 2.0, 4.0, 3.0)
                    count = detect_outliers_zscore(df, target_col, threshold)
                    st.write(f"**Outliers Found:** {count:,} ({count/len(df)*100:.2f}%)")

                elif method == "Modified Z-Score":
                    st.info("Modified Z-Score detection not implemented yet.")

                elif method == "Custom Bounds":
                    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
                    count = len(outliers)
                    st.write(f"**Outliers Found:** {count:,} ({count/len(df)*100:.2f}%)")
                    st.write(f"**Custom Bounds:** [{lower_bound}, {upper_bound}]")

            if st.button("🔧 Apply Treatment", type="primary"):
                try:
                    if method == "IQR (Recommended)":
                        outlier_info = detect_outliers_iqr(df, target_col)
                        if outlier_info:
                            _, lower, upper = outlier_info
                            if treatment == "Remove":
                                df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
                            elif treatment == "Cap to Bounds":
                                df[target_col] = np.clip(df[target_col], lower, upper)

                    elif method == "Z-Score":
                        st.warning("Z-Score based treatment is not implemented yet.")

                    elif method == "Custom Bounds":
                        if treatment == "Remove":
                            df = df[~((df[target_col] < lower_bound) | (df[target_col] > upper_bound))]
                        elif treatment == "Cap to Bounds":
                            df[target_col] = np.clip(df[target_col], lower_bound, upper_bound)

                    if treatment == "Winsorize":
                        df[target_col] = mstats.winsorize(df[target_col], limits=[0.05, 0.05])
                    elif treatment == "Log Transform":
                        df[target_col] = np.log1p(df[target_col])

                    update_df(df, f"Applied {treatment} to {target_col}")
                    st.success(f"✅ Treatment applied successfully")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab4:
        st.subheader("General Cleaning Operations")
        with st.popover("ℹ️ What are General Cleaning Operations?"):
                st.markdown(
                    """
                    General cleaning operations help you improve data quality by removing noise, redundancy, and inconsistencies that can affect analysis and modeling.

                    **Available Operations:**
                    - **Remove Constant Columns:** Removes columns where all values are the same.  
                      *These columns provide no useful information for analysis or models.*

                    - **Remove High Correlation Features:** Removes numeric columns that are very strongly related to each other.  
                      *Highly correlated columns often duplicate information and can confuse models.*

                    - **Remove Columns by Missing %:** Removes columns where too much data is missing.  
                      *Columns with a high percentage of missing values are often unreliable.*

                    - **Standardize Text Columns:** Makes text consistent by converting it to lowercase, uppercase, title case, or removing extra spaces.  
                      *This prevents the same value being treated as different due to formatting.*

                    - **Remove Special Characters:** Cleans text by removing symbols like @, #, %, or punctuation.  
                      *Useful for preparing text for analysis or modeling.*

                    - **Trim Whitespace:** Removes extra spaces before or after text values.  
                      *Helps avoid hidden duplicates caused by spacing issues.*

                    👉 These operations make your dataset cleaner, more consistent, and easier to analyze or model accurately.
                    """
                )
        
        clean_option = st.selectbox("Select Operation", [
            "Remove Constant Columns",
            "Remove High Correlation Features",
            "Remove Columns by Missing %",
            "Standardize Text Columns",
            "Remove Special Characters",
            "Trim Whitespace"
        ])
        
        if clean_option == "Remove Constant Columns":
            if st.button("🧹 Execute", type="primary"):
                constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if constant_cols:
                    df = df.drop(columns=constant_cols)
                    update_df(df, f"Removed {len(constant_cols)} constant columns")
                    st.success(f"✅ Removed: {', '.join(constant_cols)}")
                    st.rerun()
                else:
                    st.info("No constant columns found")
        
        elif clean_option == "Remove High Correlation Features":
            threshold = st.slider("Correlation Threshold", 0.8, 0.99, 0.95)
            if st.button("🧹 Execute", type="primary"):
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
                if to_drop:
                    df = df.drop(columns=to_drop)
                    update_df(df, f"Removed {len(to_drop)} high correlation columns")
                    st.success(f"✅ Removed {len(to_drop)} columns")
                    st.rerun()
                else:
                    st.info("No highly correlated columns found")
        
        elif clean_option == "Remove Columns by Missing %":
            threshold = st.slider("Missing % Threshold", 10, 90, 50)
            if st.button("🧹 Execute", type="primary"):
                missing_pct = (df.isnull().sum() / len(df)) * 100
                cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    update_df(df, f"Removed {len(cols_to_drop)} columns with >{threshold}% missing")
                    st.success(f"✅ Removed {len(cols_to_drop)} columns")
                    st.rerun()
                else:
                    st.info(f"No columns with >{threshold}% missing values")
        
        elif clean_option == "Standardize Text Columns":
            col_types = get_column_types(df)
            if col_types['categorical']:
                target_col = st.selectbox("Select column", col_types['categorical'])
                operation = st.selectbox("Operation", ["Lowercase", "Uppercase", "Title Case", "Strip Whitespace"])
                
                if st.button("🧹 Execute", type="primary"):
                    if operation == "Lowercase":
                        df[target_col] = df[target_col].str.lower()
                    elif operation == "Uppercase":
                        df[target_col] = df[target_col].str.upper()
                    elif operation == "Title Case":
                        df[target_col] = df[target_col].str.title()
                    elif operation == "Strip Whitespace":
                        df[target_col] = df[target_col].str.strip()
                    
                    update_df(df, f"Standardized {target_col}: {operation}")
                    st.success(f"✅ Applied {operation}")
                    st.rerun()


# ================================
# PAGE 5: FEATURE ENGINEERING
# ================================
elif page == "📈 Feature Engineering":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Feature Engineering")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Date/Time Features",
        "🔢 Mathematical Features",
        "🎯 Aggregation Features",
        "🔗 Interaction Features"
    ])
    
    with tab1:
        st.subheader("Date/Time Feature Engineering")
        with st.popover("ℹ️ What is Date/Time Feature Engineering?"):
                st.markdown(
                    """
                    Date/time feature engineering extracts useful information from datetime columns so models and analysis can better understand time-based patterns.

                    From a single date or timestamp, you can create multiple meaningful features such as:
                    - **Year, Month, Day:** Help capture long-term trends and seasonality.
                    - **Hour, Minute:** Useful for time-of-day patterns.
                    - **Day of Week / Day Name:** Identifies weekly cycles (e.g., weekdays vs weekends).
                    - **Week of Year / Quarter:** Captures business and seasonal reporting periods.
                    - **Is Weekend:** Flags Saturdays and Sundays.
                    - **Is Month Start / End:** Useful for financial or billing cycles.
                    - **Days in Month:** Helps normalize monthly comparisons.
                    - **Month Name:** Improves interpretability for reporting and visualization.

                    These features allow models to recognize temporal patterns that would otherwise remain hidden in raw date values.

                    👉 Date/time feature engineering improves forecasting, trend analysis, and predictive performance for time-based data.
                    """
                )
        
        col_types = get_column_types(df)
        datetime_cols = col_types['datetime'] + [col for col in df.columns 
                                                  if 'date' in col.lower() or 'time' in col.lower()]
        
        if not datetime_cols:
            st.info("No datetime columns found. Try converting a column to datetime first.")
        else:
            target_col = st.selectbox("Select datetime column", datetime_cols)
            
            # First ensure it's datetime
            if df[target_col].dtype != 'datetime64[ns]':
                if st.button("Convert to Datetime First"):
                    df[target_col] = pd.to_datetime(df[target_col], errors='coerce')
                    update_df(df, f"Converted {target_col} to datetime")
                    st.rerun()
            else:
                features_to_extract = st.multiselect("Select features to extract", [
                    "Year", "Month", "Day", "Hour", "Minute",
                    "Day of Week", "Week of Year", "Quarter",
                    "Is Weekend", "Is Month Start", "Is Month End",
                    "Days in Month", "Day Name", "Month Name"
                ])
                
                if features_to_extract and st.button("📅 Extract Features", type="primary"):
                    try:
                        if "Year" in features_to_extract:
                            df[f"{target_col}_year"] = df[target_col].dt.year
                        if "Month" in features_to_extract:
                            df[f"{target_col}_month"] = df[target_col].dt.month
                        if "Day" in features_to_extract:
                            df[f"{target_col}_day"] = df[target_col].dt.day
                        if "Hour" in features_to_extract:
                            df[f"{target_col}_hour"] = df[target_col].dt.hour
                        if "Minute" in features_to_extract:
                            df[f"{target_col}_minute"] = df[target_col].dt.minute
                        if "Day of Week" in features_to_extract:
                            df[f"{target_col}_dayofweek"] = df[target_col].dt.dayofweek
                        if "Week of Year" in features_to_extract:
                            df[f"{target_col}_weekofyear"] = df[target_col].dt.isocalendar().week
                        if "Quarter" in features_to_extract:
                            df[f"{target_col}_quarter"] = df[target_col].dt.quarter
                        if "Is Weekend" in features_to_extract:
                            df[f"{target_col}_is_weekend"] = df[target_col].dt.dayofweek.isin([5, 6]).astype(int)
                        if "Is Month Start" in features_to_extract:
                            df[f"{target_col}_is_month_start"] = df[target_col].dt.is_month_start.astype(int)
                        if "Is Month End" in features_to_extract:
                            df[f"{target_col}_is_month_end"] = df[target_col].dt.is_month_end.astype(int)
                        if "Days in Month" in features_to_extract:
                            df[f"{target_col}_days_in_month"] = df[target_col].dt.days_in_month
                        if "Day Name" in features_to_extract:
                            df[f"{target_col}_day_name"] = df[target_col].dt.day_name()
                        if "Month Name" in features_to_extract:
                            df[f"{target_col}_month_name"] = df[target_col].dt.month_name()
                        
                        update_df(df, f"Extracted {len(features_to_extract)} datetime features from {target_col}")
                        st.success(f"✅ Extracted {len(features_to_extract)} features")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Mathematical Feature Engineering")
        with st.popover("ℹ️ What is Mathematical Feature Engineering?"):
                st.markdown(
                    """
                    Mathematical feature engineering creates **new numeric columns** by applying math operations to existing numeric data.

                    👉 **Why this matters**
                    - Raw numbers alone may not show important relationships.
                    - Combining numbers mathematically helps models and analysis detect patterns, ratios, and trends.

                    👉 **Basic Operations Explained**
                    - **Addition (+):** Combines values (e.g., total cost = item price + tax)
                    - **Subtraction (-):** Finds differences (e.g., profit = revenue − cost)
                    - **Multiplication (*):** Scales values (e.g., units × price)
                    - **Division (/):** Creates ratios (e.g., cost per unit)
                    - **Power (**):** Raises one value to another (e.g., exponential growth)
                    - **Max:** Picks the larger value per row
                    - **Min:** Picks the smaller value per row
                    - **Mean:** Calculates the average of two columns

                    👉 **Polynomial Features (Simple Explanation)**
                    - Polynomial features raise a number to higher powers (square, cube, etc.).
                    - Example:
                        - Original value: x
                        - Square: x²
                        - Cube: x³
                    - These help models capture **non-linear relationships** where changes are not straight-line.

                    👉 **When to use**
                    - Ratios, differences, or totals are meaningful
                    - Growth accelerates or slows down over time
                    - You want the model to learn curved or complex patterns

                    Mathematical feature engineering improves predictive power and uncovers relationships hidden in raw numeric columns.
                    """
                )
        
        col_types = get_column_types(df)
        numeric_cols = col_types['numeric']
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns")
        else:
            st.write("**Create features from existing columns**")
            
            feat_col1, feat_col2, feat_col3 = st.columns(3)
            
            with feat_col1:
                col1 = st.selectbox("Select first column", numeric_cols, key="math_col1")
            with feat_col2:
                operation = st.selectbox("Operation", [
                    "+", "-", "*", "/", "**", "max", "min", "mean"
                ])
            with feat_col3:
                col2 = st.selectbox("Select second column", numeric_cols, key="math_col2")
            
            new_col_name = st.text_input("New column name", 
                                        value=f"{col1}_{operation}_{col2}")
            
            if st.button("➕ Create Feature", type="primary"):
                try:
                    if operation == "+":
                        df[new_col_name] = df[col1] + df[col2]
                    elif operation == "-":
                        df[new_col_name] = df[col1] - df[col2]
                    elif operation == "*":
                        df[new_col_name] = df[col1] * df[col2]
                    elif operation == "/":
                        df[new_col_name] = df[col1] / (df[col2] + 1e-10)  # Avoid division by zero
                    elif operation == "**":
                        df[new_col_name] = df[col1] ** df[col2]
                    elif operation == "max":
                        df[new_col_name] = df[[col1, col2]].max(axis=1)
                    elif operation == "min":
                        df[new_col_name] = df[[col1, col2]].min(axis=1)
                    elif operation == "mean":
                        df[new_col_name] = df[[col1, col2]].mean(axis=1)
                    
                    update_df(df, f"Created feature: {new_col_name}")
                    st.success(f"✅ Feature created")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.write("**Polynomial Features**")
            
            poly_col = st.selectbox("Select column for polynomial features", numeric_cols)
            degree = st.slider("Degree", 2, 5, 2)
            
            if st.button("🔢 Create Polynomial Features", type="primary"):
                try:
                    for d in range(2, degree + 1):
                        df[f"{poly_col}_power_{d}"] = df[poly_col] ** d
                    
                    update_df(df, f"Created polynomial features for {poly_col} up to degree {degree}")
                    st.success(f"✅ Created {degree - 1} polynomial features")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("Aggregation Features")
        with st.popover("ℹ️ What are Aggregation Features?"):
                st.markdown(
                    """
                    Aggregation features create **summary values** for each row based on a group or category.

                    👉 **What this does**
                    - Groups rows using a **categorical column** (e.g., region, customer type)
                    - Calculates a **summary statistic** from a numeric column
                    - Assigns that group-level value back to every row in the group

                    👉 **Simple Example**
                    - Group by: `Region`
                    - Aggregate: `Sales`
                    - Function: `Mean`
                    - Result: Each row shows the **average sales for its region**

                    👉 **Aggregation Functions Explained**
                    - **Mean:** Average value within the group
                    - **Median:** Middle value (less affected by outliers)
                    - **Sum:** Total value for the group
                    - **Min:** Smallest value in the group
                    - **Max:** Largest value in the group
                    - **Std:** Spread or variability within the group
                    - **Count:** Number of records in the group

                    👉 **Why this is useful**
                    - Adds **context** to individual rows
                    - Helps models compare an item against its group
                    - Captures patterns like regional averages or category behaviour

                    👉 **When to use**
                    - Customer-level data grouped by region or segment
                    - Product-level data grouped by category
                    - Time-based grouping (e.g., average sales per month)

                    Aggregation features help turn raw data into meaningful, context-aware insights.
                    """
                )
        
        col_types = get_column_types(df)
        numeric_cols = col_types['numeric']
        categorical_cols = col_types['categorical']
        
        if not numeric_cols or not categorical_cols:
            st.info("Need both numeric and categorical columns")
        else:
            agg_col1, agg_col2, agg_col3 = st.columns(3)
            
            with agg_col1:
                group_col = st.selectbox("Group by column", categorical_cols)
            with agg_col2:
                agg_col = st.selectbox("Aggregate column", numeric_cols)
            with agg_col3:
                agg_func = st.selectbox("Aggregation function", [
                    "mean", "median", "sum", "min", "max", "std", "count"
                ])
            
            if st.button("📊 Create Aggregation Feature", type="primary"):
                try:
                    if agg_func == "mean":
                        agg_result = df.groupby(group_col)[agg_col].transform('mean')
                    elif agg_func == "median":
                        agg_result = df.groupby(group_col)[agg_col].transform('median')
                    elif agg_func == "sum":
                        agg_result = df.groupby(group_col)[agg_col].transform('sum')
                    elif agg_func == "min":
                        agg_result = df.groupby(group_col)[agg_col].transform('min')
                    elif agg_func == "max":
                        agg_result = df.groupby(group_col)[agg_col].transform('max')
                    elif agg_func == "std":
                        agg_result = df.groupby(group_col)[agg_col].transform('std')
                    elif agg_func == "count":
                        agg_result = df.groupby(group_col)[agg_col].transform('count')
                    
                    new_col_name = f"{agg_col}_{agg_func}_by_{group_col}"
                    df[new_col_name] = agg_result
                    
                    update_df(df, f"Created aggregation feature: {new_col_name}")
                    st.success(f"✅ Feature created")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab4:
        st.subheader("Interaction Features")
        with st.popover("ℹ️ What are Interaction Features?"):
            st.markdown(
                """
                Interaction features combine **two columns together** to capture how they work **in combination**, not just individually.

                👉 **What to expect**
                - Shows relationships that single columns cannot explain alone
                - Helps models understand *combined effects*
                - Often improves prediction accuracy

                👉 **Numeric × Numeric Interactions**
                - Combines two numeric columns with a selected operation
                - Example:
                    - `Price + Quantity`
                    - `Price - Quantity`
                    - `Price × Quantity`
                    - `Price / Quantity`
                - Useful when the effect of one number depends on another

                👉 **Why these operations?**
                - Many real-world relationships are **not linear**
                - Interaction terms allow models to learn *joint influence*

                👉 **Categorical × Categorical Interactions**
                - Combines two categories into one new label
                - Example:
                    - `Region = North` and `Product = A`
                    - New value: `North_A`
                - Treats each combination as a unique group

                👉 **Why this matters**
                - Some category combinations behave very differently
                - Helps capture patterns like:
                    - Certain products performing better in certain regions

                👉 **When should you use interaction features**
                - Pricing, sales, demand forecasting
                - Customer behaviour analysis
                - Any case where variables influence each other

                Interaction features help models move beyond isolated variables and learn **real-world relationships** between them.
                """
            )

        col_types = get_column_types(df)
        numeric_cols = col_types['numeric']
        categorical_cols = col_types['categorical']

        st.write("**Numeric × Numeric Interactions**")
        if len(numeric_cols) >= 2:
            selected_nums = st.multiselect("Select numeric columns (2+)", numeric_cols,
                                          max_selections=5)

            operation = st.selectbox(
                "Select Operation for Numeric Interaction",
                ["Multiplication", "Addition", "Subtraction", "Division"]
            )

            if len(selected_nums) >= 2 and st.button("🔗 Create Numeric Interactions", type="primary"):
                try:
                    from itertools import combinations
                    for col1, col2 in combinations(selected_nums, 2):
                        new_col_name = f"{col1}_"
                        if operation == "Multiplication":
                            df[new_col_name + f"x_{col2}"] = df[col1] * df[col2]
                        elif operation == "Addition":
                            df[new_col_name + f"plus_{col2}"] = df[col1] + df[col2]
                        elif operation == "Subtraction":
                            df[new_col_name + f"minus_{col2}"] = df[col1] - df[col2]
                        elif operation == "Division":
                            # To avoid division by zero, replace zero with np.nan temporarily
                            df[new_col_name + f"div_{col2}"] = df[col1] / df[col2].replace(0, np.nan)

                    n_features = len(list(combinations(selected_nums, 2)))
                    update_df(df, f"Created {n_features} numeric interaction features with {operation.lower()}")
                    st.success(f"✅ Created {n_features} interaction features using {operation}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.markdown("---")
        st.write("**Categorical × Categorical Interactions**")
        if len(categorical_cols) >= 2:
            cat1 = st.selectbox("First categorical", categorical_cols, key="cat1")
            cat2 = st.selectbox("Second categorical", categorical_cols, key="cat2")

            if st.button("🔗 Create Categorical Interaction", type="primary"):
                try:
                    df[f"{cat1}_x_{cat2}"] = df[cat1].astype(str) + "_" + df[cat2].astype(str)
                    update_df(df, f"Created categorical interaction: {cat1}_x_{cat2}")
                    st.success(f"✅ Interaction feature created")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ================================
# PAGE 6: VISUALIZATIONS
# ================================
elif page == "📊 Visualizations":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Data Visualizations")
    
    col_types = get_column_types(df)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Univariate",
        "📊 Bivariate",
        "🔥 Multivariate",
        "📉 Time Series"
    ])
    
    with tab1:
        st.subheader("Univariate Analysis")
        with st.popover("ℹ️ What is Univariate Analysis?"):
            st.markdown(
                """
                Univariate analysis looks at **one variable at a time** to understand its distribution.
                """
            )

        chart_type = st.selectbox(
            "Chart Type",
            ["Histogram", "Box Plot", "Violin Plot", "Bar Chart", "Pie Chart", "Density Plot"]
        )

        if chart_type in ["Histogram", "Box Plot", "Violin Plot", "Density Plot"]:
            if not col_types["numeric"]:
                st.info("No numeric columns available")
                st.stop()
            target_col = st.selectbox("Select column", col_types["numeric"])
        else:
            if not col_types["categorical"]:
                st.info("No categorical columns available")
                st.stop()
            target_col = st.selectbox("Select column", col_types["categorical"])

        if st.button("📊 Generate Chart", type="primary"):
            try:
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=target_col, nbins=30, title=f"Histogram of {target_col}")

                elif chart_type == "Box Plot":
                    fig = px.box(df, y=target_col, title=f"Box Plot of {target_col}")

                elif chart_type == "Violin Plot":
                    fig = px.violin(df, y=target_col, box=True, points="all",
                                    title=f"Violin Plot of {target_col}")

                elif chart_type == "Density Plot":
                    fig = px.density_contour(df, x=target_col, title=f"Density Plot of {target_col}")

                elif chart_type == "Bar Chart":
                    vc = df[target_col].value_counts().head(20).reset_index()
                    vc.columns = [target_col, "count"]
                    fig = px.bar(vc, x=target_col, y="count",
                                 title=f"Bar Chart of {target_col}")

                elif chart_type == "Pie Chart":
                    vc = df[target_col].value_counts().head(10).reset_index()
                    vc.columns = [target_col, "count"]
                    fig = px.pie(vc, names=target_col, values="count",
                                 title=f"Pie Chart of {target_col}")

                st.plotly_chart(fig, use_container_width=True)

                buf = BytesIO()
                fig.write_image(buf, format="png", scale=2)
                buf.seek(0)
                st.download_button(
                    "💾 Download Chart",
                    buf,
                    f"{chart_type}_{target_col}.png",
                    "image/png"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab2:
        st.subheader("Bivariate Analysis")
        with st.popover("ℹ️ What is Bivariate Analysis?"):
            st.markdown("Relationship between **two variables**.")

        chart_type = st.selectbox(
            "Chart Type",
            ["Scatter Plot", "Line Plot", "Grouped Bar Chart", "Stacked Bar Chart", "Box Plot by Category"],
            key="bivariate_chart"
        )

        if chart_type in ["Scatter Plot", "Line Plot"]:
            if len(col_types["numeric"]) < 2:
                st.info("Need at least 2 numeric columns")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", col_types["numeric"])
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in col_types["numeric"] if c != x_col])

            hue_col = None
            if col_types["categorical"]:
                hue_col = st.selectbox("Color by (optional)", [None] + col_types["categorical"])

        elif chart_type in ["Grouped Bar Chart", "Stacked Bar Chart"]:
            if len(col_types["categorical"]) < 2:
                st.info("Need at least 2 categorical columns")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", col_types["categorical"])
            with col2:
                hue_col = st.selectbox("Group by", [c for c in col_types["categorical"] if c != x_col])

        elif chart_type == "Box Plot by Category":
            if not col_types["numeric"] or not col_types["categorical"]:
                st.info("Need numeric + categorical columns")
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                y_col = st.selectbox("Numeric column", col_types["numeric"])
            with col2:
                x_col = st.selectbox("Category", col_types["categorical"])

        if st.button("📊 Generate Chart", type="primary", key="bivariate_btn"):
            try:
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, color=hue_col,
                                     title=f"{x_col} vs {y_col}")

                elif chart_type == "Line Plot":
                    fig = px.line(df.sort_values(x_col), x=x_col, y=y_col,
                                  color=hue_col, markers=True,
                                  title=f"{x_col} vs {y_col}")

                elif chart_type in ["Grouped Bar Chart", "Stacked Bar Chart"]:
                    ct = pd.crosstab(df[x_col], df[hue_col]).reset_index()
                    melted = ct.melt(id_vars=x_col, var_name=hue_col, value_name="count")
                    fig = px.bar(
                        melted,
                        x=x_col,
                        y="count",
                        color=hue_col,
                        barmode="group" if chart_type == "Grouped Bar Chart" else "stack",
                        title=f"{chart_type}: {x_col} by {hue_col}"
                    )

                elif chart_type == "Box Plot by Category":
                    fig = px.box(df, x=x_col, y=y_col,
                                 title=f"{y_col} by {x_col}")

                st.plotly_chart(fig, use_container_width=True)

                buf = BytesIO()
                fig.write_image(buf, format="png", scale=2)
                buf.seek(0)
                st.download_button("💾 Download Chart", buf, f"{chart_type}.png", "image/png")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.subheader("Multivariate Analysis")
        with st.popover("ℹ️ What is Multivariate Analysis?"):
            st.markdown("Analysis involving **3+ variables**.")

        chart_type = st.selectbox(
            "Chart Type",
            ["Correlation Heatmap", "3D Scatter Plot"],
            key="multi_chart"
        )

        if chart_type == "Correlation Heatmap":
            numeric_cols = col_types["numeric"]

            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns")
            else:
                selected_cols = st.multiselect(
                    "Select columns (leave empty for all)",
                    numeric_cols
                )

                if st.button("🔥 Generate Heatmap", type="primary"):
                    cols = selected_cols if selected_cols else numeric_cols
                    corr = df[cols].corr()

                    fig = px.imshow(
                        corr,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap"
                    )

                    st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "3D Scatter Plot":
            if len(col_types["numeric"]) < 3:
                st.info("Need at least 3 numeric columns")
            else:
                x = st.selectbox("X-axis", col_types["numeric"])
                y = st.selectbox("Y-axis", [c for c in col_types["numeric"] if c != x])
                z = st.selectbox("Z-axis", [c for c in col_types["numeric"] if c not in [x, y]])
                color = st.selectbox("Color by (optional)", [None] + col_types["categorical"])

                if st.button("🚀 Generate 3D Plot", type="primary"):
                    fig = px.scatter_3d(
                        df,
                        x=x,
                        y=y,
                        z=z,
                        color=color,
                        title="3D Scatter Plot"
                    )

                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Time Series Visualization")
        with st.popover("ℹ️ Time Series"):
            st.markdown("Visualise numeric values over time.")

        datetime_cols = col_types["datetime"]
        numeric_cols = col_types["numeric"]

        if not datetime_cols or not numeric_cols:
            st.info("Need datetime and numeric columns")
        else:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Date column", datetime_cols)
            with col2:
                value_col = st.selectbox("Value column", numeric_cols)

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df_sorted = df.sort_values(date_col)

            chart_type = st.selectbox(
                "Chart Type",
                ["Line Plot", "Area Plot", "Rolling Average"]
            )

            if chart_type == "Rolling Average":
                window = st.slider("Rolling window", 3, 50, 7)

            if st.button("📈 Generate Time Series Plot", type="primary"):
                if chart_type == "Line Plot":
                    fig = px.line(df_sorted, x=date_col, y=value_col,
                                  title=f"Time Series: {value_col}")

                elif chart_type == "Area Plot":
                    fig = px.area(df_sorted, x=date_col, y=value_col,
                                  title=f"Area Plot: {value_col}")

                elif chart_type == "Rolling Average":
                    df_sorted["rolling"] = (
                        df_sorted.set_index(date_col)[value_col]
                        .rolling(window)
                        .mean()
                        .values
                    )

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_sorted[date_col],
                        y=df_sorted[value_col],
                        mode="lines",
                        name="Original"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_sorted[date_col],
                        y=df_sorted["rolling"],
                        mode="lines",
                        name=f"{window}-period MA"
                    ))
                    fig.update_layout(title="Rolling Average")

                st.plotly_chart(fig, use_container_width=True)


if st.session_state.show_python_editor:
    with st.expander("⚙️ Advanced Python Code Editor", expanded=True):
        st.markdown(
            """
            Use `df` to reference the dataset.  
            Pandas available as `pd`, NumPy as `np`.  
            **Example:** `df['new_col'] = df['sales'] * 1.1`  
            ⚠️ Use trusted code only.
            """
        )

        python_code = st.text_area(
            "Enter Python code",
            height=220,
            value="df['new_col'] = df['col1'] * 2"
        )

        run_python = st.button("▶ Run Python Code")
        apply_python = st.button("✅ Apply Changes")
        cancel_python = st.button("❌ Cancel")

        if run_python:
            local_vars = {"df": df.copy(), "pd": pd, "np": np}
            try:
                exec(python_code, {}, local_vars)
                st.session_state.python_preview = local_vars["df"]
                st.success("Code executed successfully. Preview below:")
                st.dataframe(st.session_state.python_preview)
            except Exception as e:
                st.error(f"Error executing code: {e}")

        if apply_python:
            if st.session_state.python_preview is not None:
                update_df(st.session_state.python_preview, "Executed custom Python code")
                st.session_state.show_python_editor = False
                st.session_state.python_preview = None
                st.rerun()
            else:
                st.warning("Run code first before applying changes.")

        if cancel_python:
            st.session_state.show_python_editor = False
            st.session_state.python_preview = None
            st.rerun()

if st.session_state.show_sql_editor:
    with st.expander("🗄️ SQL Query Editor", expanded=True):
        st.markdown(
            """
            Write SQL queries to analyze your dataset.  
            Table name: `data` (in-memory SQLite)  
            Example:  
            ```sql
            SELECT category, AVG(sales) AS avg_sales
            FROM data
            GROUP BY category
            ORDER BY avg_sales DESC;
            ```
            """
        )

        sql_code = st.text_area(
            "Enter SQL query",
            height=220,
            value="SELECT * FROM data LIMIT 10"
        )

        run_sql = st.button("▶ Run SQL Query")
        apply_sql = st.button("✅ Apply Changes")
        cancel_sql = st.button("❌ Cancel")

        if run_sql:
            try:
                conn = sqlite3.connect(":memory:")
                df.to_sql("data", conn, index=False, if_exists="replace")
                result_df = pd.read_sql(sql_code, conn)
                st.session_state.sql_preview = result_df
                st.success(f"Query executed successfully ({len(result_df)} rows)")
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"SQL Error: {e}")

        if apply_sql:
            if st.session_state.sql_preview is not None:
                update_df(st.session_state.sql_preview, "Executed custom SQL query")
                st.session_state.show_sql_editor = False
                st.session_state.sql_preview = None
                st.rerun()
            else:
                st.warning("Run query first before applying changes.")

        if cancel_sql:
            st.session_state.show_sql_editor = False
            st.session_state.sql_preview = None
            st.rerun()
# ================================
# PAGE 9: EXPORT & REPORTS
# ================================
elif page == "💾 Export & Reports":
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset first")
        st.stop()
    
    st.header("Export & Reporting")
    
    tab1, tab2, tab3 = st.tabs([
        "📤 Export Data",
        "📋 Processing Summary",
        "📊 Data Quality Report"
    ])
    
    with tab1:
        st.subheader("Export Cleaned Dataset")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON", "Parquet"])
        
        with export_col2:
            filename = st.text_input("Filename", value="cleaned_data")
        
        if st.button("📥 Download Dataset", type="primary"):
            try:
                if export_format == "CSV":
                    csv_data = df.to_csv(index=False)
                    st.download_button("⬇️ Download CSV", csv_data, f"{filename}.csv", "text/csv")
                
                elif export_format == "Excel":
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Data', index=False)
                    st.download_button("⬇️ Download Excel", buffer.getvalue(), 
                                     f"{filename}.xlsx", 
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                elif export_format == "JSON":
                    json_data = df.to_json(indent=2, orient='records')
                    st.download_button("⬇️ Download JSON", json_data, f"{filename}.json", "application/json")
                
                elif export_format == "Parquet":
                    buffer = BytesIO()
                    df.to_parquet(buffer, index=False)
                    st.download_button("⬇️ Download Parquet", buffer.getvalue(), 
                                     f"{filename}.parquet", "application/octet-stream")
                
                st.success("✅ Download ready!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Processing Summary")
        
        if "original_df" in st.session_state:
            original = st.session_state.original_df
            current = df
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Dataset:**")
                st.write(f"• Rows: {len(original):,}")
                st.write(f"• Columns: {len(original.columns):,}")
                st.write(f"• Missing values: {original.isnull().sum().sum():,}")
                st.write(f"• Duplicates: {original.duplicated().sum():,}")
                st.write(f"• Memory: {original.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
            
            with col2:
                st.write("**Current Dataset:**")
                st.write(f"• Rows: {len(current):,}")
                st.write(f"• Columns: {len(current.columns):,}")
                st.write(f"• Missing values: {current.isnull().sum().sum():,}")
                st.write(f"• Duplicates: {current.duplicated().sum():,}")
                st.write(f"• Memory: {current.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
            
            st.markdown("---")
            st.write("**Changes:**")
            row_change = len(current) - len(original)
            col_change = len(current.columns) - len(original.columns)
            
            st.write(f"• Rows: {row_change:+,}")
            st.write(f"• Columns: {col_change:+,}")
            
            st.markdown("---")
            st.write("**Processing Log:**")
            if st.session_state.processing_log:
                for log_entry in st.session_state.processing_log[-20:]:  # Last 20 actions
                    st.text(log_entry)
            else:
                st.info("No processing actions logged yet")
            
            # Export log
            if st.session_state.processing_log:
                log_text = "\n".join(st.session_state.processing_log)
                st.download_button("📥 Download Processing Log", log_text, 
                                 "processing_log.txt", "text/plain")
    
    with tab3:
        st.subheader("Data Quality Report")
        
        # Overall quality score
        quality_score, quality_dims = calculate_data_quality_score(df)
        
        st.write("**Overall Data Quality Score:**")
        
        # Progress bar for overall score
        if quality_score >= 90:
            color = "green"
        elif quality_score >= 70:
            color = "orange"
        else:
            color = "red"
        
        st.markdown(f"<h2 style='color:{color};'>{quality_score:.1f}/100</h2>", unsafe_allow_html=True)
        st.progress(quality_score / 100)
        
        st.markdown("---")
        st.write("**Quality Dimensions:**")
        
        dim_cols = st.columns(3)
        for i, (dim, score) in enumerate(quality_dims.items()):
            with dim_cols[i % 3]:
                st.metric(dim, f"{score:.1f}%")
        
        st.markdown("---")
        st.write("**Detailed Quality Checks:**")
        
        # Missing values
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            st.write(f"⚠️ **Missing Values:** {len(missing_cols)} columns affected")
            missing_df = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Count': missing_cols.values,
                'Missing %': (missing_cols.values / len(df) * 100).round(2)
            }).sort_values('Missing %', ascending=False)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No missing values")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.write(f"⚠️ **Duplicates:** {duplicates:,} rows ({duplicates/len(df)*100:.2f}%)")
        else:
            st.success("✅ No duplicates")
        
        # Outliers summary
        col_types = get_column_types(df)
        if col_types['numeric']:
            outlier_summary = []
            for col in col_types['numeric']:
                outlier_info = detect_outliers_iqr(df, col)
                if outlier_info:
                    count, _, _ = outlier_info
                    if count > 0:
                        outlier_summary.append({
                            'Column': col,
                            'Outliers': count,
                            'Outlier %': f"{(count/len(df)*100):.2f}%"
                        })
            
            if outlier_summary:
                st.write(f"⚠️ **Outliers Detected:** {len(outlier_summary)} columns")
                st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
            else:
                st.success("✅ No significant outliers detected")
        
        # Generate full report
        if st.button("📊 Generate Full HTML Report", type="primary"):
            st.info("Full HTML report generation coming soon!")



if df is None:

    st.info("👆 Upload a dataset to get started!")
