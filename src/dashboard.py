import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.scraper import IMFScraper

# Set page config
st.set_page_config(
    page_title="IMF Loans Dashboard",
    page_icon="📊",
    layout="wide"
)

def load_data():
    """Load the IMF loans data"""
    data_file = os.path.join('data', 'imf_loans.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
        print("Data loaded successfully. Columns:", df.columns.tolist())
        print("Sample data:", df.head().to_dict('records'))
        return df
    print("Warning: Data file not found at", data_file)
    return pd.DataFrame()

def run_scraper():
    """Run the scraper and return the output"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Run the scraper
        scraper = IMFScraper(data_dir='data')
        df = scraper.fetch_loans_data()
        if not df.empty:
            scraper.save_to_csv(df)
            return True, "Data scraped successfully!"
        return False, "No data was scraped. Please check the scraper implementation."
    except Exception as e:
        return False, f"Error during scraping: {str(e)}"

def main():
    st.title("IMF Loans Dashboard")
    st.markdown("Web interface for scraping and visualizing IMF loan data")
    
    # Sidebar for actions
    st.sidebar.title("Actions")
    
    # Scrape button
    if st.sidebar.button("🔄 Scrape New Data"):
        with st.spinner("Scraping data from IMF website..."):
            success, message = run_scraper()
            if success:
                st.sidebar.success(message)
                # Force a rerun to show the new data
                st.rerun()
            else:
                st.sidebar.error(message)
    
    # Display last update time
    data_file = os.path.join('data', 'imf_loans.csv')
    if os.path.exists(data_file):
        last_modified = datetime.fromtimestamp(os.path.getmtime(data_file))
        st.sidebar.caption(f"Last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    
    # Load and display data
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please click the 'Scrape New Data' button to fetch data.")
        return
    
    # Display data summary
    st.subheader("📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        total_loans = df['amount_agreed'].sum()
        st.metric("Total Loans (SDR)", f"${total_loans/1000:,.2f}B")  # Convert to billions with 2 decimal places
    with col2:
        st.metric("Number of Countries", len(df['country'].unique()))
    with col3:
        st.metric("Last Updated", df['as_of_date'].iloc[0] if 'as_of_date' in df.columns else "N/A")
    
    # Display data size info
    st.caption(f"Data contains {len(df)} records")
    
    st.markdown("---")
    
    # Convert amount to numeric
    df['amount_agreed'] = pd.to_numeric(df['amount_agreed'], errors='coerce')
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Get unique countries and sort them
    all_countries = sorted(df['country'].unique())
    
    # Debug information
    st.sidebar.write("## Debug Info")
    st.sidebar.write(f"Found {len(all_countries)} countries in the data")
    if len(all_countries) > 0:
        st.sidebar.write("Sample countries:", all_countries[:3])
    
    # Create a multi-select dropdown for country selection
    st.sidebar.markdown("### Select Countries")
    selected_countries = st.sidebar.multiselect(
        'Choose countries to display',
        options=all_countries,
        default=all_countries,  # Select all by default
        key='country_selector',
        help='Start typing to search for a country',
        placeholder='Select countries...'
    )
    
    # Apply the country filter if any countries are selected
    if selected_countries:
        df = df[df['country'].isin(selected_countries)]
    
    # Add year filter
    if 'date' in df.columns:
        # Extract year from date column
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        # Get unique years and sort them
        available_years = sorted(df['year'].unique(), reverse=True)
        
        # Add year selector to sidebar
        st.sidebar.markdown("### Filter by Year")
        selected_years = st.sidebar.multiselect(
            'Select years to include',
            options=available_years,
            default=available_years,  # Select all years by default
            key='year_selector',
            help='Filter data by year'
        )
        
        # Apply year filter
        if selected_years:
            df = df[df['year'].isin(selected_years)]
    
    # Data visualization section
    # Update the visualization title to show selected years
    if 'year' in df.columns and 'available_years' in locals() and len(selected_years) < len(available_years):
        year_range = ", ".join(map(str, sorted(selected_years)))
        st.markdown(f"### Loans by Country in Billions ({year_range})")
    else:
        st.markdown("### Loans by Country in Billions")
    st.subheader("📈 Data Visualization")
    
    # Bar chart
    st.markdown("### Loans by Country in Billions")

    # Calculate dynamic height based on number of countries
    num_countries = len(df['country'].unique())
    chart_height = max(500, num_countries * 20)  # Minimum 500px, 20px per country

    fig = px.bar(
        df.sort_values('amount_agreed', ascending=True),
        x='amount_agreed',
        y='country',
        color='country',
        labels={'amount_agreed': 'Amount (Billion USD)', 'country': 'Country'},
        height=chart_height,
        color_discrete_sequence=px.colors.qualitative.Plotly #Better color scheme
    )


    # Improve layout and readability
    fig.update_layout(
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},  # Sort by value
        xaxis_title='Amount (Billions of SDR)',
        yaxis_title='',
        margin=dict(l=150, r=50, t=50, b=50),  # Adjust margins for y-axis labels
        height=min(chart_height, 2000),  # Cap height at 2000px
        xaxis=dict(
            range=[0, df['amount_agreed'].max() * 1.1],  # Add 10% padding
            fixedrange=True  # Disable zoom/pan
        )
    )

    st.plotly_chart(fig, use_container_width=True, height=chart_height, max_height=2000)
    
    # Raw data section
    st.subheader("📋 Raw Data")
    
    # Add download button for the data
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(df)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"imf_loans_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
    )
    
    # Display the data table with search and sort
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "amount_agreed": st.column_config.NumberColumn(
                "Amount (USD)",
                format="$%,d"
            )
        },
        hide_index=True
    )

if __name__ == "__main__":
    main()
