# Top-level Streamlit setup & navigation logic
import streamlit as st
import datetime

from newsletterBuilder import create_newsletter_builder
from analytics_dashboard import create_analytics_dashboard

# Set Streamlit page config
st.set_page_config(page_title="SimPPL Newsletter & Analytics", page_icon="📰", layout="wide")

# Page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'newsletter'

# Sidebar
with st.sidebar:
    st.header("🛠 Navigation")
    page = st.radio("Select Page", ["📰 Newsletter Builder", "📊 Analytics Dashboard"], key="page_selector")
    if page == "📰 Newsletter Builder":
        st.session_state.current_page = 'newsletter'
        st.divider()
        st.subheader("Settings")
        today = datetime.date.today()
        selected_month = st.selectbox("Select Month", range(1, 13), index=today.month - 1,
                                      format_func=lambda x: datetime.date(2000, x, 1).strftime('%B'))
        selected_year = st.number_input("Select Year", min_value=2020, max_value=2035, value=today.year)
        st.session_state.selected_month = selected_month
        st.session_state.selected_year = selected_year

        if st.button("➕ Add Highlight"):
            st.session_state.highlights.append({'title': '', 'description': '', 'link': '', 'image': ''})
        if st.button("➖ Remove Highlight") and len(st.session_state.get('highlights', [])) > 1:
            st.session_state.highlights.pop()
    else:
        st.session_state.current_page = 'analytics'
        st.info("💡 View detailed analytics of your newsletter link clicks")

# Main content switch
if st.session_state.current_page == 'analytics':
    create_analytics_dashboard()
else:
    create_newsletter_builder()
