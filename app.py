import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="UIDAI Analytics 2026", layout="wide")


st.title("🛡️ UIDAI Operational Intelligence Dashboard")
st.markdown("Hackathon 2026: Data-Driven Resource Management")

# 3. File Paths
ENROLMENT_FILES = [
    './data/enrollment/api_data_aadhar_enrolment_0_500000.csv',
    './data/enrollment/api_data_aadhar_enrolment_500000_1000000.csv',
    './data/enrollment/api_data_aadhar_enrolment_1000000_1006029.csv'
]
BIOMETRIC_FILES = [
    './data/biometric/api_data_aadhar_biometric_0_500000.csv',
    './data/biometric/api_data_aadhar_biometric_500000_1000000.csv',
    './data/biometric/api_data_aadhar_biometric_1000000_1500000.csv',
    './data/biometric/api_data_aadhar_biometric_1500000_1861108.csv'
]
DEMOGRAPHIC_FILES = [
    './data/demographic/api_data_aadhar_demographic_0_500000.csv',
    './data/demographic/api_data_aadhar_demographic_500000_1000000.csv',
    './data/demographic/api_data_aadhar_demographic_1000000_1500000.csv',
    './data/demographic/api_data_aadhar_demographic_1500000_2000000.csv',
    './data/demographic/api_data_aadhar_demographic_2000000_2071700.csv'
]

# 4. Data Loading Function
@st.cache_data
def load_and_clean_data(files, type_name):
    df_list = []
    for f in files:
        try:
            temp_df = pd.read_csv(f, nrows=50000)
            df_list.append(temp_df)
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
            continue
    
    if not df_list:
        return pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['state'] = df['state'].str.strip().str.title()
    df['district'] = df['district'].str.strip().str.title()
    
    if type_name == 'Enrolment':
        df['Total'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    elif type_name == 'Biometric':
        df['Total'] = df['bio_age_5_17'] + df['bio_age_17_']
    elif type_name == 'Demographic':
        df['Total'] = df['demo_age_5_17'] + df['demo_age_17_']
    return df

# Load Data
with st.spinner("Processing Large Datasets..."):
    enrol_df = load_and_clean_data(ENROLMENT_FILES, 'Enrolment')
    bio_df = load_and_clean_data(BIOMETRIC_FILES, 'Biometric')
    demo_df = load_and_clean_data(DEMOGRAPHIC_FILES, 'Demographic')

# 5. Sidebar Filters
st.sidebar.header("Navigation Controls")
all_states = sorted(list(set(enrol_df['state']) | set(bio_df['state'])))
selected_state = st.sidebar.selectbox("Filter by State", ["All India"] + all_states)
st.sidebar.markdown("---")
st.sidebar.write("### 🚀 Team Alpha - UIDAI Hackathon 2026")
st.sidebar.info("Goal: To optimize resource allocation for Aadhar Seva Kendras using real-time data analytics.")

if selected_state != "All India":
    enrol_df = enrol_df[enrol_df['state'] == selected_state]
    bio_df = bio_df[bio_df['state'] == selected_state]
    demo_df = demo_df[demo_df['state'] == selected_state]

# 6. Top Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Total Enrolments", f"{enrol_df['Total'].sum():,}")
m2.metric("Biometric Updates", f"{bio_df['Total'].sum():,}")
m3.metric("Demographic Updates", f"{demo_df['Total'].sum():,}")

st.divider()

# 1. Health DF Calculation (Pressure Index)
h_enrol = enrol_df.groupby(['state', 'district'])['Total'].sum().reset_index()
h_bio = bio_df.groupby(['state', 'district'])['Total'].sum().reset_index()
health_df = pd.merge(h_enrol, h_bio, on=['state', 'district'], suffixes=('_Enrol', '_Bio'), how='left').fillna(0)
health_df['Pressure_Index'] = (health_df['Total_Bio'] / (health_df['Total_Enrol'] + 1)).round(2)

# 2. Anomaly Detection Logic
recent_date = enrol_df['date'].max() - pd.Timedelta(days=30)
district_recent = enrol_df[enrol_df['date'] > recent_date]
district_stats = district_recent.groupby('district')['Total'].sum().reset_index()
if not district_stats.empty:
    threshold = district_stats['Total'].mean() + 2 * district_stats['Total'].std()
    anomalies = district_stats[district_stats['Total'] > threshold]
else:
    anomalies = pd.DataFrame()

# 7. Main Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab_pincode, tab_sat, tab_action = st.tabs([
    "📍 Performance", "📈 Trends", "💡 Recommendations", 
    "🚀 Operations", "🏥 Health Check", "🔍 Raw Data", 
    "🔍 Pincode Search", "📊 Saturation Analysis", "📋 Digital Action Plan"
])

with tab1:
    st.subheader("Regional Performance")
    state_sum = enrol_df.groupby('state')['Total'].sum().nlargest(10).reset_index()
    fig1 = px.bar(state_sum, x='Total', y='state', orientation='h', title="Top 10 States", color='Total')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("Activity Trends")
    trend = enrol_df.resample('M', on='date')['Total'].sum().reset_index()
    fig2 = px.line(trend, x='date', y='Total', title="Monthly Trend Analysis")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Strategic Insights")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### High Pressure Districts")
        busy = bio_df.groupby('district')['Total'].sum().nlargest(10).reset_index()
        st.bar_chart(busy.set_index('district'))
    with col_b:
        st.write("### Newborn (0-5) Focus")
        kids = enrol_df.groupby('state')['age_0_5'].sum().nlargest(5).reset_index()
        st.plotly_chart(px.pie(kids, values='age_0_5', names='state'), use_container_width=True)

with tab4:
    st.subheader("Demand Forecasting")
    monthly_load = enrol_df.resample('M', on='date')['Total'].sum().tail(3)
    if len(monthly_load) >= 2:
        growth = (monthly_load.iloc[-1] - monthly_load.iloc[-2]) / monthly_load.iloc[-2]
        st.metric("Predicted Load Growth", f"{growth:.1%}")
        st.write("UIDAI should prepare for increased volume based on recent acceleration.")

with tab5:
    st.subheader("Service Health (Pressure Index)")
    # District-level load balancing logic
    h_enrol = enrol_df.groupby('district')['Total'].sum()
    h_bio = bio_df.groupby('district')['Total'].sum()
    pressure = (h_bio / (h_enrol + 1)).nlargest(10).reset_index()
    pressure.columns = ['District', 'Pressure Index']
    st.error("Districts requiring immediate hardware scaling:")
    st.table(pressure)

with tab6:
    st.subheader("Data Inspector")
    st.dataframe(enrol_df.head(100))

with tab_pincode:
    st.subheader("Area-specific Activity Tracker")
    
    # User se pincode input lena
    search_pincode = st.number_input("Enter 6-digit Pincode to analyze:", min_value=100000, max_value=999999, step=1)
    
    if search_pincode:
        # Pincode filter karna
        p_enrol = enrol_df[enrol_df['pincode'] == search_pincode]
        p_bio = bio_df[bio_df['pincode'] == search_pincode]
        
        if not p_enrol.empty or not p_bio.empty:
            st.success(f"Showing results for Pincode: {search_pincode}")
            
            pk1, pk2 = st.columns(2)
            pk1.metric("Local Enrolments", p_enrol['Total'].sum())
            pk2.metric("Local Biometric Updates", p_bio['Total'].sum())
            
            # Local Trend Chart
            p_trend = p_enrol.groupby('date')['Total'].sum().reset_index()
            fig_p = px.line(p_trend, x='date', y='Total', title=f"Activity Trend in {search_pincode}", markers=True)
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("No data found for this pincode. Try another one.")
            
with tab_sat:
    st.header("Enrolment vs Updates Ratio")
    st.write("This analysis helps identify regions where Aadhaar coverage is nearly 100%.")

    # 1. State-wise aggregation with safety (Handling missing states)
    comp_enrol = enrol_df.groupby('state')['Total'].sum().reset_index()
    comp_enrol.columns = ['state', 'Total_New']

    # Bio aur Demo updates ko sahi se merge karna
    bio_state = bio_df.groupby('state')['Total'].sum()
    demo_state = demo_df.groupby('state')['Total'].sum()
    
    # Dono ko add karna aur missing values ko 0 manna
    total_updates_series = bio_state.add(demo_state, fill_value=0)
    comp_updates = total_updates_series.reset_index()
    comp_updates.columns = ['state', 'Total_Updates']
    
    # 2. Final Merge and Cleaning
    comp_df = pd.merge(comp_enrol, comp_updates, on='state', how='left')
    
    # Jo states missing hain wahan Updates 0 kar dena
    comp_df = comp_df.fillna(0)
    
    # 3. Saturation Index Calculation (Safe Division)
    # Hum +1 isliye karte hain taaki Zero division error na ho
    comp_df['Saturation_Index'] = (comp_df['Total_Updates'] / (comp_df['Total_New'] + 1)).round(2)
    
    # Plotly ke liye ensure karein ki size 0 ya NaN na ho (kam se kam 1 ho)
    comp_df['Marker_Size'] = comp_df['Saturation_Index'].apply(lambda x: max(x, 1))

    # 4. Visualization
    fig_sat = px.scatter(
        comp_df, 
        x='Total_New', 
        y='Total_Updates', 
        size='Marker_Size',  # Fixed: No more NaN values
        color='state',
        hover_name='state', 
        title="Market Saturation: Enrolment vs Updates",
        labels={'Total_New': 'New Enrolments', 'Total_Updates': 'Total Updates'},
        template="plotly_white"
    )
    
    st.plotly_chart(fig_sat, use_container_width=True)
    
    st.info("💡 **Insight:** States with large bubbles but low X-axis values are 'Highly Saturated'. Resources there should be shifted from Enrolment to Update services.")
    
with tab_action:
    st.header("Strategic Directives (Automated)")
    st.write("Based on real-time data analysis, here are the top priority actions for UIDAI:")

    # 1. Resource Allocation Logic
    high_pressure_dist = health_df.nlargest(3, 'Pressure_Index')
    
    st.subheader("📍 Deployment of Mobile Vans")
    for _, row in high_pressure_dist.iterrows():
        st.warning(f"**Deploy 2 Mobile Vans to {row['district']} ({row['state']}):** Biometric update load is {row['Pressure_Index']}x higher than enrolments.")

    st.divider()

    # 2. Marketing/Campaign Logic
    low_enrol_states = enrol_df.groupby('state')['age_0_5'].sum().nsmallest(3).index.tolist()
    
    st.subheader("📢 Awareness Campaigns")
    st.info(f"**Launch 'Bal Aadhaar' Awareness in:** {', '.join(low_enrol_states)}. These states show the lowest enrolment in the 0-5 age group this month.")

    st.divider()

    # 3. Fraud/Alert Logic
    if not anomalies.empty:
        top_anomaly = anomalies.nlargest(1, 'Total').iloc[0]
        st.error(f"**Urgent Investigation:** District **{top_anomaly['district']}** showed an unusual spike of {int(top_anomaly['Total'])} activities. Audit of local Aadhaar Seva Kendras is recommended.")
    else:
        st.success("**Security Status:** No suspicious enrolment patterns detected across the region.")