import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO
import datetime

# --- Configuration ---
# Set the page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Media Intelligence Dashboard")

# Gemini API key (Canvas will inject this in a real environment)
GEMINI_API_KEY = "" # Leave empty as per instructions; Canvas will inject

# --- Helper Functions ---

# Function to normalize column names
def normalize_column_name(name):
    """Normalizes column names by converting to lowercase, replacing spaces with underscores, and stripping whitespace."""
    return name.lower().replace(' ', '_').strip()

# Function to clean data
@st.cache_data # Cache data to avoid re-running on every interaction
def clean_data(df):
    """
    Cleans the input DataFrame:
    - Converts 'date' column to datetime objects.
    - Fills missing 'engagements' with 0.
    - Normalizes column names.
    """
    # Normalize all column names first
    df.columns = [normalize_column_name(col) for col in df.columns]

    # Ensure all expected columns are present
    expected_columns = ['date', 'platform', 'sentiment', 'location', 'engagements', 'media_type']
    for col in expected_columns:
        if col not in df.columns:
            st.error(f"Missing required column: '{col}'. Please ensure your CSV has all necessary columns.")
            st.stop() # Stop execution if critical columns are missing

    # Convert 'date' to datetime, coercing errors to NaT (Not a Time)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Fill missing 'engagements' with 0
    df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0)

    # Fill missing categorical data with 'Unknown'
    for col in ['platform', 'sentiment', 'location', 'media_type']:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()

    # Drop rows where 'date' is NaT after conversion
    df.dropna(subset=['date'], inplace=True)

    return df

# Function to generate insights for charts
def generate_chart_insights(chart_title, data, chart_type):
    """Generates basic insights for a given chart based on its title and type."""
    insights = []

    if chart_title == 'Sentiment Breakdown':
        sentiment_counts = data['sentiment'].value_counts(normalize=True)
        if not sentiment_counts.empty:
            top_sentiment = sentiment_counts.index[0]
            top_percentage = (sentiment_counts.iloc[0] * 100).round(1)
            insights.append(f"The most prevalent sentiment is **{top_sentiment}**, representing {top_percentage}% of the total entries.")
            if len(sentiment_counts) > 1:
                second_sentiment = sentiment_counts.index[1]
                second_percentage = (sentiment_counts.iloc[1] * 100).round(1)
                insights.append(f"**{second_sentiment}** is the second most common sentiment at {second_percentage}%.")
            insights.append("Understanding sentiment distribution is key to tailoring communication strategies effectively.")
        else:
            insights.append("No sentiment data available for analysis.")

    elif chart_title == 'Engagement Trend over Time':
        if not data.empty:
            daily_engagements = data.groupby(data['date'].dt.date)['engagements'].sum()
            if not daily_engagements.empty:
                max_engagement_date = daily_engagements.idxmax()
                max_engagement_value = daily_engagements.max()
                min_engagement_date = daily_engagements.idxmin()
                min_engagement_value = daily_engagements.min()
                avg_daily_engagement = daily_engagements.mean().round(0)
                insights.append(f"Highest engagement was recorded on **{max_engagement_date}** with **{int(max_engagement_value)}** engagements.")
                insights.append(f"Lowest engagement was recorded on **{min_engagement_date}** with **{int(min_engagement_value)}** engagements.")
                insights.append(f"The average daily engagement across the period is approximately **{int(avg_daily_engagement)}**.")
            else:
                insights.append("No daily engagement data to show trend.")
        else:
            insights.append("No engagement data available for analysis.")

    elif chart_title == 'Platform Engagements':
        platform_eng = data.groupby('platform')['engagements'].sum().sort_values(ascending=False)
        if not platform_eng.empty:
            insights.append(f"**{platform_eng.index[0]}** is the leading platform, generating **{int(platform_eng.iloc[0])}** engagements.")
            if len(platform_eng) > 1:
                insights.append(f"The second most engaging platform is **{platform_eng.index[1]}** with **{int(platform_eng.iloc[1])}** engagements.")
            insights.append("Focusing resources on top-performing platforms can maximize campaign reach.")
        else:
            insights.append("No platform engagement data available for analysis.")

    elif chart_title == 'Media Type Mix':
        media_type_counts = data['media_type'].value_counts(normalize=True)
        if not media_type_counts.empty:
            top_media_type = media_type_counts.index[0]
            top_percentage = (media_type_counts.iloc[0] * 100).round(1)
            insights.append(f"**{top_media_type}** is the most common media type, accounting for {top_percentage}% of content.")
            if len(media_type_counts) > 1:
                second_media_type = media_type_counts.index[1]
                second_percentage = (media_type_counts.iloc[1] * 100).round(1)
                insights.append(f"**{second_media_type}** is the second most common, representing {second_percentage}%.")
            insights.append("Optimizing content formats to align with popular media types can enhance user engagement.")
        else:
            insights.append("No media type data available for analysis.")

    elif chart_title == 'Top 5 Locations by Engagements':
        location_eng = data[data['location'] != 'Unknown'].groupby('location')['engagements'].sum().nlargest(5)
        if not location_eng.empty:
            insights.append(f"The primary engagement driver by location is **{location_eng.index[0]}** with **{int(location_eng.iloc[0])}** engagements.")
            if len(location_eng) > 1:
                insights.append(f"**{location_eng.index[1]}** is the second most engaging location, contributing **{int(location_eng.iloc[1])}** engagements.")
            insights.append("Geographically targeted campaigns could yield better results in these high-engagement areas.")
        else:
            insights.append("No significant location data or all locations are 'Unknown'.")

    return insights

# Function to call Gemini API for recommendations
def get_ai_recommendations(data_summary):
    """
    Calls the Gemini API to get strategic recommendations based on provided data summary.
    """
    if not GEMINI_API_KEY:
        st.error("Gemini API key is not configured. Please set the GEMINI_API_KEY.")
        return "AI recommendations cannot be generated without an API key."

    prompt = f"""Based on the following media intelligence data, provide strategic recommendations for improving engagement and optimizing content. Focus on actionable insights across sentiment, platforms, media types, and locations. Provide the recommendations as a bulleted list in markdown format.

    Data Summary:
    {data_summary}

    Recommendations:
    """

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})

    payload = {"contents": chat_history}
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and result["candidates"][0]["content"]["parts"][0].get("text"):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Failed to get recommendations from AI. Response structure unexpected."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with AI: {e}. Please try again later."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Streamlit UI Layout ---

st.title("Interactive Media Intelligence Dashboard")

# 1. Upload CSV File & Date Filter
st.header("1. Upload CSV File & Date Filter")
st.write("Please upload a CSV file with these columns: `Date`, `Platform`, `Sentiment`, `Location`, `Engagements`, `Media Type`.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date (Optional)", value=None)
with col2:
    end_date = st.date_input("End Date (Optional)", value=None)

# Store processed data in session state
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = pd.DataFrame()
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = pd.DataFrame()

process_button = st.button("Process Data")
clear_button = st.button("Clear/Reset Dashboard")

if clear_button:
    st.session_state.df_cleaned = pd.DataFrame()
    st.session_state.df_filtered = pd.DataFrame()
    st.session_state.ai_recommendations = "" # Clear AI recommendations
    uploaded_file = None # Reset file uploader display
    st.success("Dashboard cleared!")
    st.rerun() # Rerun to clear all cached data and UI

if process_button and uploaded_file is not None:
    # Read the CSV file
    string_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
    try:
        df_raw = pd.read_csv(string_data)
        st.session_state.df_cleaned = clean_data(df_raw.copy()) # Store cleaned data
        st.success("CSV file processed successfully!")

        # Apply date filtering
        df_filtered_by_date = st.session_state.df_cleaned.copy()
        if start_date:
            df_filtered_by_date = df_filtered_by_date[df_filtered_by_date['date'] >= pd.Timestamp(start_date)]
        if end_date:
            df_filtered_by_date = df_filtered_by_date[df_filtered_by_date['date'] <= pd.Timestamp(end_date) + pd.Timedelta(days=1, seconds=-1)] # End of day

        if df_filtered_by_date.empty:
            st.warning("No data matches the selected date range.")
            st.session_state.df_filtered = pd.DataFrame()
        else:
            st.session_state.df_filtered = df_filtered_by_date
            st.success(f"Data filtered by date range. {len(st.session_state.df_filtered)} records selected.")

    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        st.session_state.df_cleaned = pd.DataFrame()
        st.session_state.df_filtered = pd.DataFrame()

# 2. Data Processing and Cleaning (Implicit in the process button logic)
st.header("2. Data Processing and Cleaning")
st.write("""
The uploaded data will be automatically cleaned:
- 'Date' column will be converted to datetime objects.
- Missing values in 'Engagements' will be filled with 0.
- Column names will be normalized for consistency.
- Data will be filtered by the selected date range.
""")

# Display processed data table
if not st.session_state.df_filtered.empty:
    st.header("3. Processed Data Table")
    st.dataframe(st.session_state.df_filtered.reset_index(drop=True), use_container_width=True)

# 4. Interactive Charts & 5. Top Insights
st.header("4. Interactive Charts & 5. Top Insights")

if not st.session_state.df_filtered.empty:
    # Sentiment Breakdown
    st.subheader("Sentiment Breakdown")
    sentiment_counts = st.session_state.df_filtered['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment', title='Sentiment Breakdown',
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_chart_insights("Sentiment Breakdown", st.session_state.df_filtered, 'pie'):
        st.markdown(f"- {insight}")

    # Engagement Trend over Time
    st.subheader("Engagement Trend over Time")
    daily_engagements = st.session_state.df_filtered.groupby(st.session_state.df_filtered['date'].dt.date)['engagements'].sum().reset_index()
    daily_engagements.columns = ['Date', 'Engagements']
    fig_engagement_trend = px.line(daily_engagements, x='Date', y='Engagements', title='Engagement Trend over Time')
    st.plotly_chart(fig_engagement_trend, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_chart_insights("Engagement Trend over Time", st.session_state.df_filtered, 'line'):
        st.markdown(f"- {insight}")

    # Platform Engagements
    st.subheader("Platform Engagements")
    platform_eng = st.session_state.df_filtered.groupby('platform')['engagements'].sum().reset_index()
    platform_eng.columns = ['Platform', 'Total_Engagements']
    platform_eng = platform_eng.sort_values('Total_Engagements', ascending=False)
    fig_platform_eng = px.bar(platform_eng, x='Platform', y='Total_Engagements', title='Platform Engagements',
                              color_discrete_sequence=px.colors.qualitative.D3)
    st.plotly_chart(fig_platform_eng, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_chart_insights("Platform Engagements", st.session_state.df_filtered, 'bar'):
        st.markdown(f"- {insight}")

    # Media Type Mix
    st.subheader("Media Type Mix")
    media_type_counts = st.session_state.df_filtered['media_type'].value_counts().reset_index()
    media_type_counts.columns = ['Media_Type', 'Count']
    fig_media_type = px.pie(media_type_counts, values='Count', names='Media_Type', title='Media Type Mix',
                            color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_media_type, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_chart_insights("Media Type Mix", st.session_state.df_filtered, 'pie'):
        st.markdown(f"- {insight}")

    # Top 5 Locations
    st.subheader("Top 5 Locations by Engagements")
    location_eng = st.session_state.df_filtered[st.session_state.df_filtered['location'] != 'Unknown'] \
                   .groupby('location')['engagements'].sum().nlargest(5).reset_index()
    location_eng.columns = ['Location', 'Total_Engagements']
    fig_top_locations = px.bar(location_eng, x='Location', y='Total_Engagements', title='Top 5 Locations by Engagements',
                               color_discrete_sequence=px.colors.qualitative.Dark2)
    st.plotly_chart(fig_top_locations, use_container_width=True)
    st.markdown("**Top 3 Insights:**")
    for insight in generate_chart_insights("Top 5 Locations by Engagements", st.session_state.df_filtered, 'bar'):
        st.markdown(f"- {insight}")

    # 6. AI-Powered Data Recommendations
    st.header("6. AI-Powered Data Recommendations")
    st.write("Click the button below to get strategic recommendations generated by AI based on the processed data.")

    # Summarize data for AI prompt
    data_summary_for_ai = ""
    data_summary_for_ai += "- **Sentiment Breakdown:** " + ", ".join([f"{k}: {v}" for k, v in st.session_state.df_filtered['sentiment'].value_counts().to_dict().items()]) + "\n"

    daily_eng = st.session_state.df_filtered.groupby(st.session_state.df_filtered['date'].dt.date)['engagements'].sum()
    if not daily_eng.empty:
        max_eng = daily_eng.max()
        min_eng = daily_eng.min()
        avg_eng = daily_eng.mean().round(0)
        max_date = daily_eng.idxmax()
        min_date = daily_eng.idxmin()
        data_summary_for_ai += f"- **Engagement Trend (Daily):** Highest: {int(max_eng)} on {max_date}, Lowest: {int(min_eng)} on {min_date}, Average Daily: {int(avg_eng)}\n"
    else:
        data_summary_for_ai += f"- **Engagement Trend (Daily):** No data\n"

    platform_eng = st.session_state.df_filtered.groupby('platform')['engagements'].sum().sort_values(ascending=False)
    data_summary_for_ai += "- **Platform Engagements:** " + ", ".join([f"{k}: {int(v)}" for k, v in platform_eng.to_dict().items()]) + "\n"

    media_type_counts = st.session_state.df_filtered['media_type'].value_counts()
    data_summary_for_ai += "- **Media Type Mix:** " + ", ".join([f"{k}: {v}" for k, v in media_type_counts.to_dict().items()]) + "\n"

    location_eng = st.session_state.df_filtered[st.session_state.df_filtered['location'] != 'Unknown'].groupby('location')['engagements'].sum().nlargest(5)
    data_summary_for_ai += "- **Top 5 Locations by Engagements:** " + ", ".join([f"{k}: {int(v)}" for k, v in location_eng.to_dict().items()]) + "\n"


    if st.button("Generate AI Recommendations"):
        with st.spinner("Generating AI recommendations..."):
            ai_output = get_ai_recommendations(data_summary_for_ai)
            st.session_state.ai_recommendations = ai_output # Store AI recommendations in session state
            st.subheader("AI Recommendations:")
            st.markdown(st.session_state.ai_recommendations)
    elif st.session_state.ai_recommendations: # Display if already generated
        st.subheader("AI Recommendations:")
        st.markdown(st.session_state.ai_recommendations)

else:
    st.info("Upload a CSV file and click 'Process Data' to view charts and insights.")

