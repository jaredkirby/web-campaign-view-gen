import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path


# Import necessary components from the original script
from campaign_report import (
    read_and_clean_data,
    load_historical_data,
    find_changes,
    save_historical_data,
    generate_reports,
    send_reports_by_email,
    EmailConfig,
    CampaignReportEmailer,
    setup_logging,
    categorize_campaigns,
    format_budget,
)

# Configure page settings
st.set_page_config(
    page_title="Campaign Report Generator", page_icon="📊", layout="wide"
)


def initialize_app():
    """Initialize the Streamlit application state"""
    # Setup logging
    setup_logging()

    # Initialize session state variables
    if "processed_df" not in st.session_state:
        st.session_state.processed_df = None
    if "report_paths" not in st.session_state:
        st.session_state.report_paths = None
    if "email_sent" not in st.session_state:
        st.session_state.email_sent = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None


def create_temp_directory():
    """Create a temporary directory for report generation"""
    if st.session_state.temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        (temp_dir / "campaign_history").mkdir(exist_ok=True)
        st.session_state.temp_dir = temp_dir
    return st.session_state.temp_dir


def display_campaign_section(campaigns: pd.DataFrame, section_title: str):
    """Display a section of campaigns with expandable details"""
    if not campaigns.empty:
        total_budget = campaigns["Tactic Allocated Budget"].sum()
        campaign_count = len(campaigns)
        changes_count = len([c for c in campaigns["changes"] if c])

        st.markdown(f"### {section_title} ({campaign_count} Campaigns)")
        st.markdown(f"**Total Budget: {format_budget(total_budget)}**")

        if changes_count > 0:
            st.info(f"{changes_count} campaigns have changes")

        # Group by retailer
        for retailer in sorted(campaigns["Retailer"].unique()):
            retailer_campaigns = campaigns[campaigns["Retailer"] == retailer]
            retailer_budget = retailer_campaigns["Tactic Allocated Budget"].sum()

            with st.expander(f"{retailer} - {format_budget(retailer_budget)}"):
                for _, campaign in retailer_campaigns.iterrows():
                    display_campaign_details(campaign)


def display_campaign_details(campaign: pd.Series):
    """Display detailed information for a single campaign"""
    changes = campaign.get("changes", [])
    is_new = changes == ["New Campaign"]
    has_changes = bool(changes) and not is_new

    title = f"{campaign['Tactic Brand']} - {campaign['Event Name']}"
    if is_new:
        title = "🆕 " + title
    elif has_changes:
        title = "⚠️ " + title

    st.markdown(f"**{title}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"- Start Date: {campaign['Tactic Start Date'].strftime('%Y-%m-%d')}"
        )
        st.markdown(f"- End Date: {campaign['Tactic End Date'].strftime('%Y-%m-%d')}")
        st.markdown(f"- Budget: {format_budget(campaign['Tactic Allocated Budget'])}")
    with col2:
        st.markdown(f"- Product: {campaign['Tactic Product']}")
        st.markdown(f"- Order ID: {campaign['Tactic Order ID']}")
        if campaign["Tactic Vendor"]:
            st.markdown(f"- Vendor: {campaign['Tactic Vendor']}")

    if has_changes:
        with st.expander("View Changes"):
            for change in changes:
                st.markdown(f"- {change}")


def display_data_preview(df: pd.DataFrame):
    """Display a preview of the processed campaign data"""
    st.subheader("Campaign Overview")

    # Categorize campaigns
    current_campaigns, future_campaigns, past_campaigns = categorize_campaigns(df)

    # Display summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Campaigns", len(df))
    with col2:
        total_budget = df["Tactic Allocated Budget"].sum()
        st.metric("Total Budget", format_budget(total_budget))
    with col3:
        st.metric("Active Campaigns", len(current_campaigns))
    with col4:
        st.metric("Upcoming Campaigns", len(future_campaigns))
    with col5:
        changes_count = len([c for c in df["changes"] if c])
        st.metric("Changes Detected", changes_count)

    # Display campaign sections
    tab1, tab2, tab3 = st.tabs(
        ["Active Campaigns", "Upcoming Campaigns", "Past Campaigns"]
    )

    with tab1:
        display_campaign_section(current_campaigns, "Currently Active Campaigns")
    with tab2:
        display_campaign_section(future_campaigns, "Upcoming Campaigns")
    with tab3:
        display_campaign_section(past_campaigns, "Past Campaigns")


def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file and generate reports"""
    try:
        temp_dir = create_temp_directory()

        # Save uploaded file to temp directory
        temp_csv = temp_dir / uploaded_file.name
        with open(temp_csv, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the data
        df = read_and_clean_data(temp_csv)
        historical_df = load_historical_data(temp_dir / "campaign_history")
        df = find_changes(df, historical_df)

        # Save historical data
        save_historical_data(df, temp_dir / "campaign_history")

        # Generate reports
        md_path, email_path = generate_reports(
            df,
            temp_dir / "campaign_history",
            temp_dir,
            cleanup_days=None,  # Disable cleanup in web interface
        )

        # Update session state
        st.session_state.processed_df = df
        st.session_state.report_paths = (md_path, email_path)

        return True

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logging.error(f"File processing error: {e}", exc_info=True)
        return False


def send_email_reports(
    sender_email: str, sender_password: str, primary_recipients: str, cc_recipients: str
):
    """Send the generated reports via email"""
    if not st.session_state.report_paths:
        st.error("No reports available to send")
        return False

    try:
        # Update environment variables with user input
        os.environ["EMAIL_SENDER"] = sender_email
        os.environ["EMAIL_SENDER_PASSWORD"] = sender_password
        os.environ["EMAIL_PRIMARY_RECIPIENTS"] = primary_recipients
        os.environ["EMAIL_CC_RECIPIENTS"] = cc_recipients

        md_path, email_path = st.session_state.report_paths
        success = send_reports_by_email(md_path, email_path)

        if success:
            st.session_state.email_sent = True
            st.success("Reports sent successfully!")

            st.info(
                f"Reports sent to:\n"
                f"From: {sender_email}\n"
                f"To: {primary_recipients}\n"
                f"CC: {cc_recipients}"
            )
        else:
            st.error("Failed to send reports")

        return success

    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        logging.error(f"Email sending error: {e}", exc_info=True)
        return False


def main():
    """Main Streamlit application"""
    initialize_app()

    st.title("Campaign Report Generator")

    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", help="Upload the campaign data CSV file"
    )

    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                if process_uploaded_file(uploaded_file):
                    st.success("File processed successfully!")

    # Display results if data is processed
    if st.session_state.processed_df is not None:
        st.markdown("---")

        # Email configuration
        st.subheader("Email Configuration")

        # Sender credentials
        col1, col2 = st.columns(2)
        with col1:
            sender_email = st.text_input("Sender Email", value="Taylor@cemm.com")
        with col2:
            sender_password = st.text_input("Email Password", type="password")

        # Email recipients
        col3, col4 = st.columns(2)
        with col3:
            primary_recipients = st.text_input(
                "Primary Recipients (comma-separated)", value="Rachel@cemm.com"
            )
        with col4:
            cc_recipients = st.text_input(
                "CC Recipients (comma-separated)",
                value="Jared@cemm.com, Mary@cemm.com, Roxy@cemm.com",
            )

        # Send email button
        if st.button("Send Reports", type="primary"):
            if not sender_password:
                st.error("Please enter the sender email password")
            else:
                with st.spinner("Sending reports..."):
                    send_email_reports(
                        sender_email, sender_password, primary_recipients, cc_recipients
                    )

        # Data preview with improved organization
        st.markdown("---")
        display_data_preview(st.session_state.processed_df)

        # Report preview in expandable section
        st.markdown("---")
        with st.expander("View Report Preview", expanded=False):
            tab1, tab2 = st.tabs(["Markdown Report", "Email Report"])

            with tab1:
                if st.session_state.report_paths:
                    with open(st.session_state.report_paths[0], "r") as f:
                        st.markdown(f.read())

            with tab2:
                if st.session_state.report_paths:
                    with open(st.session_state.report_paths[1], "r") as f:
                        st.text(f.read())


if __name__ == "__main__":
    main()
