import streamlit as st
import pandas as pd
import plotly.express as px

# Upload CSV
st.sidebar.header("📂 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
    return df

if uploaded_file:
    df = load_data(uploaded_file)
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# Dashboard Title
st.title("💳 Merchant Transaction Analytics Dashboard")

# Key Metrics
total_transactions = df['occurrences'].sum()
total_amount = df['total_amount'].sum()
avg_transaction_value = total_amount / total_transactions if total_transactions else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Total Amount Processed", f"${total_amount:,.2f}")
col3.metric("Avg. Transaction Value", f"${avg_transaction_value:,.2f}")

# Filters
st.sidebar.header("🔍 Filters")
selected_merchant = st.sidebar.multiselect("Select Merchants", df['merchant_name'].unique())
selected_device = st.sidebar.multiselect("Select Devices", df['device_name'].unique())

# Filter data
filtered_df = df.copy()
if selected_merchant:
    filtered_df = filtered_df[filtered_df['merchant_name'].isin(selected_merchant)]
if selected_device:
    filtered_df = filtered_df[filtered_df['device_name'].isin(selected_device)]

# Main Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Merchant Analysis", "Device Insights", "Card Patterns", "Raw Data"])

with tab1:
    st.subheader("Merchant Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        merchant_trans = filtered_df.groupby('merchant_name')['occurrences'].sum().nlargest(10)
        fig = px.bar(merchant_trans, title="Top Merchants by Transactions", color=merchant_trans.index)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        merchant_amt = filtered_df.groupby('merchant_name')['total_amount'].sum().nlargest(10)
        fig = px.pie(merchant_amt, names=merchant_amt.index, title="Revenue Distribution by Merchant")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Device Analysis")
    device_usage = filtered_df.groupby('device_name')['occurrences'].sum()
    fig = px.treemap(device_usage, path=[device_usage.index], values=device_usage.values, title="Transaction Volume by Device")
    st.plotly_chart(fig, use_container_width=True)

import statsmodels.api as sm
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

with tab3:
    # Get the selected merchants and display them in the title
    selected_merchants_str = ', '.join(selected_merchant) if selected_merchant else "All Merchants"
    st.subheader(f"Card Activity for {selected_merchants_str}")
    
    # Filter the data based on the selected merchants
    filtered_cards = filtered_df[filtered_df['merchant_name'].isin(selected_merchant)] if selected_merchant else filtered_df

    # Calculate the number of distinct cards for the selected merchants
    distinct_cards_count = filtered_cards['masked_card_no'].nunique()

    # Add an input field for the user to specify the number of cards to display
    num_cards_to_show = st.number_input(
        "Enter number of cards to display",
        min_value=1, 
        max_value=distinct_cards_count,  # Set max_value to the number of distinct cards
        value=10
    )

    # Group by masked_card_no and merchant_name to aggregate occurrences and total_amount
    card_activity = filtered_cards.groupby(['masked_card_no', 'merchant_name']).agg({'occurrences': 'sum', 'total_amount': 'sum'}).reset_index()

    # Sort and select the cards based on the number of occurrences
    selected_cards = card_activity.nlargest(num_cards_to_show, 'occurrences')

    # Clustering (KMeans)
    X = selected_cards[['occurrences', 'total_amount']]  # Selecting the features for clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters (e.g., 3 clusters)
    selected_cards['cluster'] = kmeans.fit_predict(X)  # Fit and predict clusters

    # Scatter plot with clusters
    fig = px.scatter(
        selected_cards, x='total_amount', y='occurrences', size='total_amount', color='cluster',
        title=f"Cards: Frequency vs Amount (Clusters)",
        labels={'cluster': 'Cluster'},
        color_continuous_scale='Viridis'  # Optional: Adjust the color scale for better visualization
    )

    # Provide a unique key for the plotly_chart
    st.plotly_chart(fig, use_container_width=True, key=f"card_activity_{selected_merchant}")

    # Show cluster centers
    st.write("Cluster Centers (Centroids):")
    st.write(kmeans.cluster_centers_)

    # Optionally, display the cards and their assigned clusters
    st.write("Cards with Cluster Assignments:")
    st.dataframe(selected_cards[['masked_card_no', 'merchant_name', 'occurrences', 'total_amount', 'cluster']])



with tab4:
    st.subheader("Transaction Data")
    st.dataframe(filtered_df.sort_values('total_amount', ascending=False))
