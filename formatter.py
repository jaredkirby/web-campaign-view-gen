# formatter.py

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from pathlib import Path


@dataclass
class CampaignMetrics:
    """Store campaign metrics for consistent display across views"""

    total_campaigns: int
    total_budget: float
    active_campaigns: int
    upcoming_campaigns: int
    past_campaigns: int
    changes_count: int
    new_campaigns: int


@dataclass
class RetailerGroup:
    """Store retailer-specific campaign data"""

    retailer: str
    budget: float
    campaigns: pd.DataFrame


@dataclass
class MonthGroup:
    """Store month-specific campaign data"""

    month: str
    budget: float
    retailers: List[RetailerGroup]


class CampaignDisplayFormatter:
    """Single source of truth for formatting campaign data across different views"""

    def __init__(self, df: pd.DataFrame, campaign_type: str = "offsite"):
        self.df = df
        self.campaign_type = campaign_type
        self.current_date = pd.Timestamp.now().normalize()

        # Categorize campaigns once for reuse
        self.current_campaigns, self.future_campaigns, self.past_campaigns = (
            self._categorize_campaigns()
        )

        # Calculate metrics once for reuse
        self.metrics = self._calculate_metrics()

    def _categorize_campaigns(self):
        """Categorize campaigns into current, future, and past"""
        # Convert Tactic Start Date and End Date to pandas Timestamp if they aren't already
        self.df["Tactic Start Date"] = pd.to_datetime(self.df["Tactic Start Date"])
        self.df["Tactic End Date"] = pd.to_datetime(self.df["Tactic End Date"])

        current = self.df[
            (self.df["Tactic Start Date"] <= self.current_date)
            & (self.df["Tactic End Date"] >= self.current_date)
        ].copy()

        future = self.df[self.df["Tactic Start Date"] > self.current_date].copy()
        past = self.df[self.df["Tactic End Date"] < self.current_date].copy()

        # Sort each category
        current.sort_values(
            ["Tactic End Date", "Retailer", "Tactic Brand"], inplace=True
        )
        future.sort_values(
            ["Tactic Start Date", "Retailer", "Tactic Brand"], inplace=True
        )
        past.sort_values(
            ["Tactic End Date", "Retailer", "Tactic Brand"],
            ascending=[False, True, True],
            inplace=True,
        )

        return current, future, past

    def _calculate_metrics(self) -> CampaignMetrics:
        """Calculate metrics for display"""
        return CampaignMetrics(
            total_campaigns=len(self.df),
            total_budget=self.df["Tactic Allocated Budget"].sum(),
            active_campaigns=len(self.current_campaigns),
            upcoming_campaigns=len(self.future_campaigns),
            past_campaigns=len(self.past_campaigns),
            changes_count=len([c for c in self.df["changes"] if c]),
            new_campaigns=len([c for c in self.df["changes"] if c == ["New Campaign"]]),
        )

    def _group_by_retailer(self, df: pd.DataFrame) -> List[RetailerGroup]:
        """Group campaigns by retailer"""
        retailer_groups = []
        for retailer in sorted(df["Retailer"].unique()):
            retailer_df = df[df["Retailer"] == retailer]
            retailer_groups.append(
                RetailerGroup(
                    retailer=retailer,
                    budget=retailer_df["Tactic Allocated Budget"].sum(),
                    campaigns=retailer_df,
                )
            )
        return retailer_groups

    def _group_by_month(self, df: pd.DataFrame) -> List[MonthGroup]:
        """Group campaigns by month and retailer"""
        df = df.copy()
        df["Month"] = df["Tactic Start Date"].dt.strftime("%B %Y")
        month_groups = []

        for month in sorted(df["Month"].unique()):
            month_df = df[df["Month"] == month]
            month_groups.append(
                MonthGroup(
                    month=month,
                    budget=month_df["Tactic Allocated Budget"].sum(),
                    retailers=self._group_by_retailer(month_df),
                )
            )
        return month_groups

    def format_campaign_details(self, campaign: pd.Series) -> Dict:
        """Format individual campaign details consistently"""
        is_new = campaign.get("changes", []) == ["New Campaign"]
        has_changes = bool(campaign.get("changes", [])) and not is_new

        details = {
            "title": f"{campaign['Tactic Brand']} - {campaign['Tactic Name']}",
            "is_new": is_new,
            "has_changes": has_changes,
            "retailer": campaign["Retailer"],
            "brand": campaign["Tactic Brand"],
            "product": campaign["Tactic Product"],
            "start_date": campaign["Tactic Start Date"].strftime("%Y-%m-%d"),
            "end_date": campaign["Tactic End Date"].strftime("%Y-%m-%d"),
            "budget": campaign["Tactic Allocated Budget"],
            "order_id": campaign["Tactic Order ID"],
            "changes": campaign.get("changes", []),
        }

        # Add optional fields
        for field in ["Tactic Vendor", "Tactic Description", "Event Name", "Event ID"]:
            if field in campaign and not pd.isna(campaign[field]):
                details[field.lower()] = campaign[field]

        # Handle aggregated campaigns for onsite campaigns only when multiple sub-lines exist
        if (
            self.campaign_type == "onsite"
            and isinstance(campaign.get("Sub_Lines"), list)
            and len(campaign.get("Sub_Lines", [])) > 1
        ):  # Only mark as aggregated if multiple sub-lines
            details["is_aggregated"] = True
            details["sub_lines"] = campaign["Sub_Lines"]
            budget_types = campaign.get("Budget_Types", [])
            details["budget_types"] = (
                budget_types if isinstance(budget_types, list) else []
            )
        else:
            details["is_aggregated"] = False

        return details

    def get_section_data(self, section: str) -> Dict:
        """Get formatted data for a specific section"""
        df_map = {
            "current": self.current_campaigns,
            "future": self.future_campaigns,
            "past": self.past_campaigns,
        }

        df = df_map[section]

        if section == "future":
            return {
                "month_groups": self._group_by_month(df),
                "total_budget": df["Tactic Allocated Budget"].sum(),
                "campaign_count": len(df),
                "changes_count": len([c for c in df["changes"] if c]),
            }
        else:
            return {
                "retailer_groups": self._group_by_retailer(df),
                "total_budget": df["Tactic Allocated Budget"].sum(),
                "campaign_count": len(df),
                "changes_count": len([c for c in df["changes"] if c]),
            }

    def generate_markdown(self, file_path: Path) -> None:
        """Generate markdown report using consistent formatting"""
        with open(file_path, "w", encoding="utf-8") as f:
            # Write header and summary
            f.write(
                f"# Campaign Status Report - {self.current_date.strftime('%Y-%m-%d')}\n\n"
            )
            self._write_markdown_summary(f)

            # Write sections
            for section, title in [
                ("current", "Currently Active Campaigns"),
                ("future", "Upcoming Campaigns"),
                ("past", "Completed Campaigns"),
            ]:
                self._write_markdown_section(f, section, title)

    def generate_email(self, file_path: Path) -> None:
        """Generate email report using consistent formatting"""
        with open(file_path, "w", encoding="utf-8") as f:
            # Write header and summary
            f.write(
                f"Campaign Status Report - {self.current_date.strftime('%Y-%m-%d')}\n"
            )
            f.write("=" * 80 + "\n\n")
            self._write_email_summary(f)

            # Write sections
            for section, title in [
                ("current", "CURRENTLY ACTIVE CAMPAIGNS"),
                ("future", "UPCOMING CAMPAIGNS"),
                ("past", "COMPLETED CAMPAIGNS"),
            ]:
                self._write_email_section(f, section, title)

    def _write_markdown_summary(self, f):
        """Write summary section in markdown format"""
        f.write("## Summary\n")
        if self.metrics.changes_count > 0:
            f.write(
                f"**🔄 Changes Detected: {self.metrics.changes_count} campaigns updated**\n"
            )
        if self.metrics.new_campaigns > 0:
            f.write(f"**🆕 New Campaigns: {self.metrics.new_campaigns}**\n")

        f.write(f"- Currently Active Campaigns: {self.metrics.active_campaigns}\n")
        f.write(f"- Upcoming Campaigns: {self.metrics.upcoming_campaigns}\n")
        f.write(f"- Completed Campaigns: {self.metrics.past_campaigns}\n")
        f.write(
            f"- Total Budget Across All Campaigns: ${self.metrics.total_budget:,.2f}\n\n"
        )

        if self.metrics.changes_count > 0:
            f.write("### Change Indicators:\n")
            f.write("- ⚠️ Campaign has changes\n")
            f.write("- 🆕 New campaign\n")
            f.write("- 🔄 Number of changes in section\n\n")

        f.write("---\n\n")

    def _write_email_summary(self, f):
        """Write summary section in email format"""
        f.write("SUMMARY\n-------\n")
        if self.metrics.changes_count > 0:
            f.write(
                f"Changes Detected: {self.metrics.changes_count} campaigns updated\n"
            )
        if self.metrics.new_campaigns > 0:
            f.write(f"New Campaigns: {self.metrics.new_campaigns}\n")

        f.write(f"Currently Active Campaigns: {self.metrics.active_campaigns}\n")
        f.write(f"Upcoming Campaigns: {self.metrics.upcoming_campaigns}\n")
        f.write(f"Completed Campaigns: {self.metrics.past_campaigns}\n")
        f.write(
            f"Total Budget Across All Campaigns: ${self.metrics.total_budget:,.2f}\n\n"
        )
        f.write("=" * 80 + "\n\n")

    def _write_markdown_section(self, f, section: str, title: str):
        """Write section in markdown format"""
        data = self.get_section_data(section)

        # Write section header
        changes_indicator = (
            f" (🔄 {data['changes_count']} with changes)"
            if data["changes_count"]
            else ""
        )
        f.write(f"# {title} ({data['campaign_count']} Campaigns{changes_indicator})\n")
        f.write(f"**Total Budget: ${data['total_budget']:,.2f}**\n\n")

        if section == "future":
            for month_group in data["month_groups"]:
                f.write(f"## {month_group.month} (${month_group.budget:,.2f})\n\n")
                for retailer in month_group.retailers:
                    self._write_markdown_retailer_group(f, retailer)
        else:
            for retailer in data["retailer_groups"]:
                self._write_markdown_retailer_group(f, retailer)

    def _write_markdown_retailer_group(self, f, retailer: RetailerGroup):
        """Write retailer group in markdown format"""
        f.write(f"### {retailer.retailer} (${retailer.budget:,.2f})\n\n")
        for _, campaign in retailer.campaigns.iterrows():
            details = self.format_campaign_details(campaign)
            self._write_markdown_campaign(f, details)

    def _write_markdown_campaign(self, f, details: Dict):
        """Write individual campaign in markdown format"""
        change_indicator = "⚠️ " if details["has_changes"] else ""
        f.write(f"- **{change_indicator}{details['retailer']}** - {details['brand']}\n")

        if details["is_new"]:
            f.write("  - 🆕 **New Campaign**\n")

        # Write common details
        f.write(f"  - Product: {details['product']}\n")
        f.write(f"  - Campaign: {details['title']}\n")

        # Write optional details
        for field in ["tactic_vendor", "tactic_description"]:
            if field in details:
                f.write(
                    f"  - {field.replace('tactic_', '').title()}: {details[field]}\n"
                )

        # Write dates and budget
        if details["is_aggregated"]:
            f.write(f"  - Total Budget: ${details['budget']:,.2f}\n")
            f.write("  - Campaign Schedule:\n")
            for sub in details["sub_lines"]:
                f.write(f"    - {sub['tactic_name']} (Order ID: {sub['order_id']})\n")
                f.write(
                    f"      Start Date: {sub['start_date']}, Budget: ${sub['budget']:,.2f} ({sub['budget_type']})\n"
                )
        else:
            f.write(f"  - Dates: {details['start_date']} to {details['end_date']}\n")
            f.write(f"  - Budget: ${details['budget']:,.2f}\n")
            f.write(f"  - Order ID: {details['order_id']}\n")

        # Write changes if any
        if details["has_changes"]:
            f.write("  - **Changes Detected:**\n")
            for change in details["changes"]:
                f.write(f"    - {change}\n")

        f.write("\n")

    def _write_email_section(self, f, section: str, title: str):
        """Write section in email format"""
        data = self.get_section_data(section)
        indent = "  "

        # Write section header
        f.write(f"{title} ({data['campaign_count']} Campaigns)\n")
        if data["changes_count"] > 0:
            f.write(f"{data['changes_count']} campaigns have changes\n")
        f.write(f"Total Budget: ${data['total_budget']:,.2f}\n\n")

        if section == "future":
            # For future campaigns, organize by month
            for month_group in data["month_groups"]:
                f.write(f"{month_group.month} (${month_group.budget:,.2f})\n\n")

                for retailer in month_group.retailers:
                    f.write(
                        f"{indent}{retailer.retailer} (${retailer.budget:,.2f})\n\n"
                    )

                    for _, campaign in retailer.campaigns.iterrows():
                        self._write_email_campaign(f, campaign, indent_level=2)
                    f.write("\n")
        else:
            # For current and past campaigns, organize by retailer
            for retailer in data["retailer_groups"]:
                f.write(f"{retailer.retailer} (${retailer.budget:,.2f})\n\n")

                for _, campaign in retailer.campaigns.iterrows():
                    self._write_email_campaign(f, campaign)

        f.write("-" * 80 + "\n\n")

    def _write_email_campaign(self, f, campaign: pd.Series, indent_level: int = 1):
        """Write individual campaign in email format"""
        details = self.format_campaign_details(campaign)
        indent = "  " * indent_level

        # Add indicators for new or changed campaigns
        indicators = []
        if details["is_new"]:
            indicators.append("[NEW]")
        elif details["has_changes"]:
            indicators.append("[UPDATED]")
        indicator_str = " ".join(indicators) + " " if indicators else ""

        # Write basic info
        f.write(f"{indent}{indicator_str}{details['retailer']} - {details['brand']}\n")
        f.write(f"{indent}Product: {details['product']}\n")
        f.write(f"{indent}Campaign: {details['title']}\n")

        # Write optional fields
        if "event_name" in details:
            f.write(f"{indent}Event: {details['event_name']}\n")
        if "event_id" in details:
            event_link = f"https://cemm-ajinomoto.shopperationsapp.com/events/{details['event_id']}/spend"
            f.write(
                f"{indent}Event ID: {details['event_id']} (<a href='{event_link}'>View in Shopperations</a>)\n"
            )
        if "tactic_vendor" in details:
            f.write(f"{indent}Vendor: {details['tactic_vendor']}\n")

        # Handle aggregated campaigns differently
        if details["is_aggregated"]:
            f.write(
                f"{indent}Combined Campaign Group: {details['title'].split(' - ')[1]}\n"
            )
            f.write(f"{indent}Total Combined Budget: ${details['budget']:,.2f}\n")
            f.write(f"{indent}Budget Sources: {', '.join(details['budget_types'])}\n")
            f.write(f"{indent}Sub-campaigns:\n")

            for sub in details["sub_lines"]:
                f.write(
                    f"{indent}  - {sub['tactic_name']} (Order ID: {sub['order_id']})\n"
                )
                f.write(f"{indent}    Start Date: {sub['start_date']}\n")
                f.write(
                    f"{indent}    Budget: ${sub['budget']:,.2f} ({sub['budget_type']})\n"
                )
        else:
            f.write(
                f"{indent}Dates: {details['start_date']} to {details['end_date']}\n"
            )
            f.write(f"{indent}Budget: ${details['budget']:,.2f}\n")
            f.write(f"{indent}Order ID: {details['order_id']}\n")

        # Write changes if any
        if details["has_changes"]:
            f.write(f"{indent}Changes:\n")
            for change in details["changes"]:
                f.write(f"{indent}  * {change}\n")

        f.write("\n")

    def _write_email_summary(self, f):
        """Write summary section in email format"""
        f.write("SUMMARY\n-------\n")
        if self.metrics.changes_count > 0:
            f.write(
                f"Changes Detected: {self.metrics.changes_count} campaigns updated\n"
            )
        if self.metrics.new_campaigns > 0:
            f.write(f"New Campaigns: {self.metrics.new_campaigns}\n")

        f.write(f"Currently Active Campaigns: {self.metrics.active_campaigns}\n")
        f.write(f"Upcoming Campaigns: {self.metrics.upcoming_campaigns}\n")
        f.write(f"Completed Campaigns: {self.metrics.past_campaigns}\n")
        f.write(
            f"Total Budget Across All Campaigns: ${self.metrics.total_budget:,.2f}\n\n"
        )
        f.write("=" * 80 + "\n\n")

    def _write_markdown_section(self, f, section: str, title: str):
        """Write section in markdown format"""
        data = self.get_section_data(section)

        # Write section header
        changes_indicator = (
            f" (🔄 {data['changes_count']} with changes)"
            if data["changes_count"]
            else ""
        )
        f.write(f"# {title} ({data['campaign_count']} Campaigns{changes_indicator})\n")
        f.write(f"**Total Budget: ${data['total_budget']:,.2f}**\n\n")

        if section == "future":
            for month_group in data["month_groups"]:
                f.write(f"## {month_group.month} (${month_group.budget:,.2f})\n\n")
                for retailer in month_group.retailers:
                    self._write_markdown_retailer_group(f, retailer)
        else:
            for retailer in data["retailer_groups"]:
                self._write_markdown_retailer_group(f, retailer)

    def _write_markdown_retailer_group(self, f, retailer: RetailerGroup):
        """Write retailer group in markdown format"""
        f.write(f"### {retailer.retailer} (${retailer.budget:,.2f})\n\n")
        for _, campaign in retailer.campaigns.iterrows():
            details = self.format_campaign_details(campaign)
            self._write_markdown_campaign(f, details)

    def _write_markdown_campaign(self, f, details: Dict):
        """Write individual campaign in markdown format"""
        change_indicator = "⚠️ " if details["has_changes"] else ""
        f.write(f"- **{change_indicator}{details['retailer']}** - {details['brand']}\n")

        if details["is_new"]:
            f.write("  - 🆕 **New Campaign**\n")

        # Write common details
        f.write(f"  - Product: {details['product']}\n")
        f.write(f"  - Campaign: {details['title']}\n")

        # Write optional details
        for field in ["tactic_vendor", "tactic_description"]:
            if field in details:
                f.write(
                    f"  - {field.replace('tactic_', '').title()}: {details[field]}\n"
                )

        # Write dates and budget
        if details["is_aggregated"]:
            f.write(f"  - Total Budget: ${details['budget']:,.2f}\n")
            f.write("  - Campaign Schedule:\n")
            for sub in details["sub_lines"]:
                f.write(f"    - {sub['tactic_name']} (Order ID: {sub['order_id']})\n")
                f.write(
                    f"      Start Date: {sub['start_date']}, Budget: ${sub['budget']:,.2f} ({sub['budget_type']})\n"
                )
        else:
            f.write(f"  - Dates: {details['start_date']} to {details['end_date']}\n")
            f.write(f"  - Budget: ${details['budget']:,.2f}\n")
            f.write(f"  - Order ID: {details['order_id']}\n")

        # Write changes if any
        if details["has_changes"]:
            f.write("  - **Changes Detected:**\n")
            for change in details["changes"]:
                f.write(f"    - {change}\n")

        f.write("\n")
