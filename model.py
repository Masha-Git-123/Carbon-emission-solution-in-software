import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Vehicle Data Dashboard", layout="wide")
st.title("🚛 Vehicle Data Dashboard (No AI Mode)")
st.markdown("Upload your dataset. The buttons below run standard Python formulas to analyze the data.")

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Drop your CSV file here", type=["csv"])

# --- MAIN APP LOGIC ---
if uploaded_file is not None:
    # 1. Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Clean column names (strip spaces to avoid errors)
        df.columns = df.columns.str.strip()
        
        st.write("### 📊 Data Preview")
        st.dataframe(df.head(3))
        
        st.divider()

        # --- PREDEFINED QUESTIONS ---
        st.write("### ⚡ Select an Analysis")
        col1, col2, col3 = st.columns(3)

        # --- LOGIC FOR BUTTON 1: HEAVIEST CARGO ---
        if col1.button("🏆 Find Heaviest Cargo"):
            st.info("Analyzing: **Which company carries the heaviest Cargo Weight?**")
            
            # Pure Python Logic: Find the row with max Cargo Weight
            try:
                max_cargo_row = df.loc[df['Cargo Weight (kg)'].idxmax()]
                company_name = max_cargo_row['Company']
                weight = max_cargo_row['Cargo Weight (kg)']
                
                st.success(f"Answer: **{company_name}**")
                st.write(f"They carry **{weight} kg**, which is the highest in the list.")
            except KeyError:
                st.error("Error: Column 'Cargo Weight (kg)' not found in CSV.")

        # --- LOGIC FOR BUTTON 2: CALCULATE CO2 ---
        if col2.button("💰 Calculate CO2 Costs"):
            st.info("Analyzing: **Calculate Total Monthly CO2 Emissions for each company.**")
            
            # Pure Python Logic: Multiply columns
            try:
                # Calculate Total CO2 = Trips * Emission per Trip
                df['Total Monthly Emission'] = df['Trips per Month'] * df['CO2 Emission per Trip (kg)']
                
                # Create a clean table to show
                result = df[['Company', 'Total Monthly Emission']].sort_values(by='Total Monthly Emission', ascending=False)
                
                st.success("Calculation Complete!")
                st.dataframe(result)
            except KeyError:
                st.error("Error: Check your column names. Need 'Trips per Month' and 'CO2 Emission per Trip (kg)'.")

        # --- LOGIC FOR BUTTON 3: PLOT CHART ---
        if col3.button("📈 Chart: Truck vs Van"):
            st.info("Analyzing: **Comparing Average Distance of Trucks vs Vans**")
            
            try:
                # Pure Python Logic: Group by Vehicle Type and take the Average
                avg_dist = df.groupby('Vehicle Type')['Average Distance (km)'].mean()
                
                # Plotting
                fig, ax = plt.subplots()
                colors = ['#ff9999','#66b3ff'] # Red and Blue
                avg_dist.plot(kind='bar', ax=ax, color=colors)
                
                ax.set_title("Average Distance by Vehicle Type")
                ax.set_ylabel("Distance (km)")
                
                st.pyplot(fig)
            except KeyError:
                st.error("Error: Column 'Vehicle Type' or 'Average Distance (km)' not found.")

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("👈 Waiting for file... Please upload your CSV in the sidebar.")