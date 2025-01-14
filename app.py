import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


# Import necessary components from the original script
from campaign_report import (
    read_and_clean_data,
    load_historical_data,
    find_changes,
    save_historical_data,
    generate_reports,
    aggregate_onsite_campaigns,
    EmailConfig,
    CampaignReportEmailer,
    setup_logging,
    format_budget,
    cleanup_old_reports,
    get_unique_filename,
)
from formatter import CampaignDisplayFormatter, RetailerGroup, MonthGroup

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
    if "campaign_type" not in st.session_state:
        st.session_state.campaign_type = "offsite"


def create_temp_directory():
    """Create a temporary directory for report generation"""
    if st.session_state.temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        (temp_dir / "campaign_history").mkdir(exist_ok=True)
        st.session_state.temp_dir = temp_dir
    return st.session_state.temp_dir


def display_campaign_section(
    campaigns: pd.DataFrame, section_title: str, campaign_type: str
):
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
                    display_campaign_details(campaign, campaign_type)


def display_campaign_details(campaign_details: Dict, campaign_type: str):
    """Display detailed information for a single campaign"""
    # Format title with indicators
    title = campaign_details["title"]
    if campaign_details["is_new"]:
        title = "🆕 " + title
    elif campaign_details["has_changes"]:
        title = "⚠️ " + title
    if campaign_details["is_aggregated"]:
        title += " (Combined Campaign)"

    st.markdown(f"**{title}**")

    col1, col2 = st.columns(2)

    with col1:
        if campaign_details["is_aggregated"]:
            st.markdown("**Campaign Schedule:**")
            for sub_line in campaign_details["sub_lines"]:
                st.markdown(
                    f"- **{sub_line['tactic_name']}** (Order ID: {sub_line['order_id']})"
                )
                st.markdown(f"  - Start: {sub_line['start_date']}")
                st.markdown(
                    f"  - Budget: ${sub_line['budget']:,.2f} ({sub_line['budget_type']})"
                )
            st.markdown("---")
            st.markdown(
                f"**Total Combined Budget: ${campaign_details['budget']:,.2f}**"
            )
        else:
            st.markdown(f"- Start Date: {campaign_details['start_date']}")
            st.markdown(f"- End Date: {campaign_details['end_date']}")
            st.markdown(f"- Budget: ${campaign_details['budget']:,.2f}")
            st.markdown(f"- Order ID: {campaign_details['order_id']}")

    with col2:
        st.markdown(f"- Product: {campaign_details['product']}")
        if "tactic_vendor" in campaign_details:
            st.markdown(f"- Vendor: {campaign_details['tactic_vendor']}")

        # Show budget types for aggregated campaigns
        if campaign_details["is_aggregated"] and campaign_details.get("budget_types"):
            st.markdown("- **Budget Types:**")
            for budget_type in campaign_details["budget_types"]:
                st.markdown(f"  - {budget_type}")

    if campaign_details["has_changes"]:
        with st.expander("View Changes"):
            for change in campaign_details["changes"]:
                st.markdown(f"- {change}")

    st.markdown("---")  # Add separator between campaigns


def display_retailer_group(
    retailer_group: RetailerGroup,
    campaign_formatter: CampaignDisplayFormatter,
    campaign_type: str,
):
    """Display a group of campaigns for a retailer"""
    with st.expander(f"{retailer_group.retailer} - ${retailer_group.budget:,.2f}"):
        for _, campaign in retailer_group.campaigns.iterrows():
            display_campaign_details(
                campaign_formatter.format_campaign_details(campaign), campaign_type
            )


def display_month_group(
    month_group: MonthGroup,
    campaign_formatter: CampaignDisplayFormatter,
    campaign_type: str,
):
    """Display a group of campaigns for a month"""
    st.markdown(f"## {month_group.month} (${month_group.budget:,.2f})")

    for retailer_group in month_group.retailers:
        display_retailer_group(retailer_group, campaign_formatter, campaign_type)


def display_data_preview(formatter: CampaignDisplayFormatter):
    """Display a preview of the processed campaign data using the formatter"""
    st.subheader("Campaign Overview")

    # Display summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Campaigns", formatter.metrics.total_campaigns)
    with col2:
        st.metric("Total Budget", f"${formatter.metrics.total_budget:,.2f}")
    with col3:
        st.metric("Active Campaigns", formatter.metrics.active_campaigns)
    with col4:
        st.metric("Upcoming Campaigns", formatter.metrics.upcoming_campaigns)
    with col5:
        st.metric("Changes Detected", formatter.metrics.changes_count)

    # Display campaign sections in tabs
    tab1, tab2, tab3 = st.tabs(
        ["Active Campaigns", "Upcoming Campaigns", "Past Campaigns"]
    )

    with tab1:
        current_data = formatter.get_section_data("current")
        st.markdown(
            f"### Currently Active Campaigns ({current_data['campaign_count']} Campaigns)"
        )
        st.markdown(f"**Total Budget: ${current_data['total_budget']:,.2f}**")
        if current_data["changes_count"] > 0:
            st.info(f"{current_data['changes_count']} campaigns have changes")
        for retailer_group in current_data["retailer_groups"]:
            display_retailer_group(retailer_group, formatter, formatter.campaign_type)

    with tab2:
        future_data = formatter.get_section_data("future")
        st.markdown(
            f"### Upcoming Campaigns ({future_data['campaign_count']} Campaigns)"
        )
        st.markdown(f"**Total Budget: ${future_data['total_budget']:,.2f}**")
        if future_data["changes_count"] > 0:
            st.info(f"{future_data['changes_count']} campaigns have changes")
        for month_group in future_data["month_groups"]:
            display_month_group(month_group, formatter, formatter.campaign_type)

    with tab3:
        past_data = formatter.get_section_data("past")
        st.markdown(f"### Past Campaigns ({past_data['campaign_count']} Campaigns)")
        st.markdown(f"**Total Budget: ${past_data['total_budget']:,.2f}**")
        if past_data["changes_count"] > 0:
            st.info(f"{past_data['changes_count']} campaigns have changes")
        for retailer_group in past_data["retailer_groups"]:
            display_retailer_group(retailer_group, formatter, formatter.campaign_type)


def generate_reports(
    formatter: CampaignDisplayFormatter, output_dir: Path, cleanup_days: Optional[int]
) -> Tuple[Path, Path]:
    """Generate both markdown and email reports using the formatter"""
    # Clean up old reports if requested
    if cleanup_days is not None:
        cleanup_old_reports(output_dir, cleanup_days)

    # Create output filenames
    timestamp = datetime.now().strftime("%Y%m%d")
    base_md_path = (
        output_dir
        / f"{formatter.campaign_type.capitalize()}_Campaign_Status_Report_{timestamp}.md"
    )
    base_email_path = (
        output_dir
        / f"{formatter.campaign_type.capitalize()}_Campaign_Status_Email_{timestamp}.txt"
    )

    md_path = get_unique_filename(base_md_path)
    email_path = get_unique_filename(base_email_path)

    # Generate reports
    formatter.generate_markdown(md_path)
    formatter.generate_email(email_path)

    return md_path, email_path


def process_uploaded_file(uploaded_file, campaign_type: str):
    """Process the uploaded CSV file with enhanced error handling"""
    try:
        temp_dir = create_temp_directory()

        # Save uploaded file to temp directory
        temp_csv = temp_dir / uploaded_file.name
        with open(temp_csv, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Process the data
        df = read_and_clean_data(temp_csv)
        logging.info(f"Data after cleaning - shape: {df.shape}")
        logging.info(f"Columns after cleaning: {df.columns.tolist()}")

        if len(df) == 0:
            raise ValueError("No valid data rows after cleaning")

        # Apply aggregation for onsite campaigns
        if campaign_type == "onsite":
            logging.info("Starting onsite aggregation")
            df = aggregate_onsite_campaigns(df)
            logging.info(f"Data after aggregation - shape: {df.shape}")

        # Process historical data and changes
        historical_df = load_historical_data(temp_dir / "campaign_history")
        df = find_changes(df, historical_df)

        # Save historical data
        save_historical_data(df, temp_dir / "campaign_history")

        # Create formatter instance
        formatter = CampaignDisplayFormatter(df, campaign_type)

        # Generate reports
        md_path, email_path = generate_reports(
            formatter, temp_dir, None  # cleanup_days
        )

        # Update session state
        st.session_state.processed_df = df
        st.session_state.report_paths = (md_path, email_path)
        st.session_state.campaign_type = campaign_type
        st.session_state.formatter = formatter

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

        # Create EmailConfig directly
        config = EmailConfig(
            sender_email=sender_email,
            primary_recipients=primary_recipients.split(","),
            cc_recipients=cc_recipients.split(",") if cc_recipients else [],
        )

        # Create emailer with configuration and password
        emailer = CampaignReportEmailer(config, sender_password)

        # Send the report
        success = emailer.send_campaign_report(
            md_path, email_path, datetime.now().strftime("%Y-%m-%d")
        )

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

    # Campaign type selection
    campaign_type = st.radio(
        "Select Campaign Type",
        options=["offsite", "onsite"],
        format_func=lambda x: x.capitalize(),
        help="Select the type of campaign data you are processing",
    )

    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", help="Upload the campaign data CSV file"
    )

    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file..."):
                if process_uploaded_file(uploaded_file, campaign_type):
                    st.success("File processed successfully!")

    # Display results if data is processed
    if st.session_state.processed_df is not None and hasattr(
        st.session_state, "formatter"
    ):
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
        display_data_preview(st.session_state.formatter)

        # Report preview in expandable section
        st.markdown("---")
        with st.expander("View Report Preview"):
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
