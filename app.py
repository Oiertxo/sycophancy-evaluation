import streamlit as st
import pandas as pd
import json

# Page configuration
st.set_page_config(page_title="Sycophancy Dataset Explorer", layout="wide")

@st.cache_data
def load_data(file_path):
    """Loads the JSONL dataset and converts it to a Pandas DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# --- Data Loading ---
try:
    # Adjust the path to your curated_dataset.jsonl
    df = load_data('data/curated_dataset.jsonl')
except FileNotFoundError:
    st.error("File 'data/curated_dataset.jsonl' not found. Please check the path.")
    st.stop()

# --- Header Section ---
st.title("🧪 Sycophancy Multi-Bias Explorer")
st.markdown(f"Exploring **{len(df)}** interactions across **{df['group'].nunique()}** base scenarios.")

# --- Top Filter Area (No nan allowed) ---
st.write("### 🎯 Global Filters")

# .dropna() removes the null/nan values from the filter list
all_biases = df['bias_type'].dropna().unique().tolist()

selected_biases = st.multiselect(
    "Filter Induced Variations by Bias Type:", 
    options=all_biases, 
    default=all_biases,
    help="The Control prompt is always visible. Use this to filter the variations shown below."
)

st.divider()

# --- Main Layout: Navigation and Content ---
col_nav, col_view = st.columns([1, 2])

with col_nav:
    st.subheader("📁 Groups / Scenarios")
    # Get the first 'basic_situation' for each group to build the labels
    group_options = df.groupby('group')['basic_situation'].first().reset_index()
    group_list = [f"Group {row['group']}: {row['basic_situation']}" for _, row in group_options.iterrows()]
    
    # Navigation Radio Buttons
    selected_group_str = st.radio(
        "Select a scenario to inspect:", 
        group_list,
        index=0
    )
    selected_group_id = int(selected_group_str.split(":")[0].split(" ")[1])

# --- Data Processing ---
group_df = df[df['group'] == selected_group_id]

# Control item is identified by having a null bias_type
control_item = group_df[group_df['bias_type'].isna()].iloc[0]

# Induced items must match the selected biases and NOT be null
induced_items = group_df[group_df['bias_type'].isin(selected_biases)]

with col_view:
    st.subheader("🔍 Scenario Details")
    
    # 1. DISPLAY CONTROL
    st.info(f"**CONTROL PROMPT (Original)**\n\n{control_item['dilemma_situation']}")
    
    st.divider()
    
    # 2. DISPLAY VARIATIONS
    st.write(f"### Induced Variations ({len(induced_items)})")
    
    if induced_items.empty:
        st.warning("No variations match the selected filters.")
    
    for _, item in induced_items.iterrows():
        label = f"ID {item['idx']} | Bias: {item['bias_type']} | Outcome: {item['outcome']}"
        with st.expander(label):
            original = control_item['dilemma_situation']
            induced = item['dilemma_situation']
            
            # Smart Diff: If the bias is just an addition at the end
            if induced.startswith(original):
                bias_part = induced[len(original):]
                st.write("**Base Text:**")
                st.text(original)
                st.write("**Bias Injection:**")
                st.success(bias_part)
            else:
                st.write("**Full Prompt:**")
                st.write(induced)

# --- Master Table (Bottom) ---
st.divider()
st.subheader("📊 Raw Data View")
# We filter the master table by bias type but keep it sorted by index
master_table_view = df[df['bias_type'].isin(selected_biases) | df['bias_type'].isna()]
st.dataframe(master_table_view, use_container_width=True)