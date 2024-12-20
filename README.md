# Campaign Report Generator

A Streamlit web application for generating and sending campaign status reports. This tool processes campaign data from CSV files, generates detailed reports, and sends them via email to specified recipients.

## Features

-   **CSV Data Processing**: Upload and process campaign data files
-   **Interactive Data Preview**: View campaign details organized by status (Active, Upcoming, Past)
-   **Change Detection**: Automatically identifies and highlights changes in campaign data
-   **Report Generation**: Creates formatted reports in both Markdown and email-friendly formats
-   **Email Integration**: Send reports directly through the application with configurable recipients

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd web-campaign-view-gen
```

2. Install required dependencies:

```bash
pip install streamlit pandas
```

## Usage

1. Start the Streamlit application:

```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Using the application:
    - Upload your campaign data CSV file
    - Click "Process File" to analyze the data
    - Configure email settings:
        - Sender email (default: Taylor@cemm.com)
        - Email password
        - Primary recipients (default: Rachel@cemm.com)
        - CC recipients (default: Jared@cemm.com, Mary@cemm.com, Roxy@cemm.com)
    - Review the campaign overview and reports
    - Click "Send Reports" to distribute via email

## CSV File Requirements

Your input CSV file must include the following columns:

-   Tactic Start Date
-   Tactic End Date
-   Tactic Vendor
-   Retailer
-   Tactic Brand
-   Event Name
-   Tactic Name
-   Tactic Description
-   Tactic Product
-   Tactic Order ID
-   Event ID
-   Tactic Allocated Budget

## Features Details

### Campaign Overview

-   Total campaign count and budget
-   Active, upcoming, and past campaign metrics
-   Changes detection and highlighting
-   Detailed campaign information organized by retailer

### Report Types

1. **Markdown Report**: Comprehensive formatted report with detailed campaign information
2. **Email Report**: Simplified format optimized for email viewing

### Email Configuration

-   Configurable sender email and password
-   Multiple recipient support (Primary and CC)
-   Success confirmation and delivery status

## Troubleshooting

Common issues and solutions:

1. **File Upload Errors**

    - Ensure your CSV file contains all required columns
    - Check for proper date formatting in the CSV

2. **Email Sending Issues**
    - Verify sender email and password are correct
    - Ensure all email addresses are properly formatted
    - Check for proper network connectivity
