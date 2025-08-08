"""
Meta-Learning MVP CLI Application

A command-line interface for meta-learning with classifier ensembles
using latent space.
Provides functionality to manage datasets, find best classifiers,
and create ensembles using latent space.
"""

import logging

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from app.config.config import settings
from app.database.models import KnowledgeBaseRepository
from app.logger_config import setup_logger
from app.services.knowledge_builder_service import KnowledgeBuilderService

console = Console()


def setup() -> None:
    """Setup for application"""
    try:
        setup_logger()
        logging.info(
            "Starting CLI app '%s' version '%s'",
            settings.project_name,
            settings.project_version,
        )
    except AttributeError as ex:
        logging.error("Failed to provision application on start: %s", ex)


class MetaLearningCLI:
    """Main CLI application class"""

    def __init__(self):
        self.console = Console()
        self.knowledge_builder_service = KnowledgeBuilderService()
        self.repository = KnowledgeBaseRepository()

    def show_welcome(self) -> None:
        """Show welcome message and application info"""
        welcome_text = Text()
        welcome_text.append("🧠 Meta-Learning MVP CLI\n", style="bold blue")
        welcome_text.append(f"Version: {settings.project_version}\n", style="dim")
        welcome_text.append(
            "A tool for meta-learning with classifier ensembles using OpenML datasets",
            style="italic",
        )

        panel = Panel(
            welcome_text, title="Welcome", border_style="blue", padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

    def show_menu(self) -> None:
        """Display the main menu options"""
        # Get knowledge base summary stats
        stats = self.knowledge_builder_service.get_base_learner_summary()

        menu_text = Text()
        menu_text.append("Available Actions:\n\n", style="bold")
        menu_text.append(
            "1. 🔄 Update Knowledge Base (⚠️  CLEARS ALL DATA)\n", style="cyan"
        )
        menu_text.append("2. 📊 Inspect Knowledge Base\n", style="magenta")
        menu_text.append("3. ❌ Exit\n", style="red")

        if stats and not stats.get("error"):
            menu_text.append("\nCurrent Knowledge Base:\n", style="bold dim")
            menu_text.append(
                f"  • Total entries: {stats.get('total_entries', 0)}\n", style="dim"
            )
            menu_text.append(
                f"  • Unique algorithms: {stats.get('unique_algorithms', 0)}\n",
                style="dim",
            )
            menu_text.append(
                f"  • Datasets covered: {stats.get('datasets_covered', 0)}\n",
                style="dim",
            )

        panel = Panel(
            menu_text, title="Main Menu", border_style="green", padding=(1, 2)
        )
        self.console.print(panel)

    def inspect_knowledge_base(self) -> None:
        """Display knowledge base entries with pagination"""
        try:
            page_size = 10
            current_page = 0
            total_entries = self.repository.count_entries()
            total_pages = (
                (total_entries + page_size - 1) // page_size if total_entries > 0 else 0
            )

            if total_entries == 0:
                self.console.print(
                    "📭 No entries found in knowledge base.", style="yellow"
                )
                return

            while True:
                entries = self._get_page_entries(page_size, current_page)
                if not entries:
                    self.console.print("📄 No more entries to display.", style="dim")
                    current_page = max(0, current_page - 1)
                    continue

                self._display_entries_table(
                    entries, current_page, total_pages, total_entries
                )
                choice = self._show_pagination_controls(current_page, total_pages)

                if choice == "n" and current_page < total_pages - 1:
                    current_page += 1
                elif choice == "p" and current_page > 0:
                    current_page -= 1
                elif choice == "d":
                    self._show_entry_details(entries)
                elif choice == "q":
                    break

                self.console.print()

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.console.print(f"❌ Error inspecting knowledge base: {e}", style="red")

    def _get_page_entries(self, page_size: int, current_page: int):
        """Get entries for the current page"""
        offset = current_page * page_size
        return self.repository.get_all(limit=page_size, offset=offset)

    def _display_entries_table(
        self, entries, current_page: int, total_pages: int, total_entries: int
    ) -> None:
        """Display entries in a table format"""
        title = (
            f"Knowledge Base Entries "
            f"(Page {current_page + 1} of {total_pages}, "
            f"{total_entries} total)"
        )
        table = Table(title=title)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Run ID", style="blue", no_wrap=True)
        table.add_column("Algorithm", style="green")
        table.add_column("Dataset", style="magenta")
        table.add_column("Metric", style="yellow", no_wrap=True)
        table.add_column("Score", style="red", justify="right")
        table.add_column("Created", style="dim", no_wrap=True)

        for entry in entries:
            self._add_entry_row(table, entry)

        self.console.print(table)

    def _add_entry_row(self, table: Table, entry) -> None:
        """Add a single entry row to the table"""
        created_str = (
            entry.created_at.strftime("%Y-%m-%d %H:%M") if entry.created_at else "N/A"
        )

        primary_metric, primary_score = self._get_primary_metric_display(entry)

        table.add_row(
            str(entry.id) if entry.id else "N/A",
            str(entry.run_id),
            (
                entry.flow_name[:30] + "..."
                if len(entry.flow_name) > 30
                else entry.flow_name
            ),
            (
                entry.data_name[:20] + "..."
                if len(entry.data_name) > 20
                else entry.data_name
            ),
            primary_metric,
            primary_score,
            created_str,
        )

    def _get_primary_metric_display(self, entry):
        """Get primary metric name and score for display"""
        if not entry.metrics:
            return "N/A", "N/A"

        preferred_metrics = [
            "predictive_accuracy",
            "area_under_roc_curve",
            "f_measure",
        ]

        metric_name = None
        for pref_metric in preferred_metrics:
            if pref_metric in entry.metrics:
                metric_name = pref_metric
                break

        if not metric_name:
            metric_name = next(iter(entry.metrics.keys()))

        return metric_name, f"{entry.metrics[metric_name]:.4f}"

    def _show_pagination_controls(self, current_page: int, total_pages: int):
        """Show pagination controls and get user choice"""
        self.console.print()
        controls = Text()
        controls.append("Controls: ", style="bold")

        choices = []
        if current_page < total_pages - 1:
            controls.append("[n]ext page • ", style="cyan")
            choices.append("n")
        if current_page > 0:
            controls.append("[p]revious page • ", style="cyan")
            choices.append("p")

        controls.append("[d]etails for entry • ", style="magenta")
        controls.append("[q]uit", style="red")
        choices.extend(["d", "q"])

        self.console.print(controls)
        choice = Prompt.ask("Action", choices=choices, default="q").lower()
        return choice

    def _show_entry_details(
        self, entries: list
    ) -> None:  # pylint: disable=too-many-branches
        """Show detailed information for a specific entry"""
        try:
            entry_id = IntPrompt.ask(
                "Enter entry ID for details",
                default=entries[0].id if entries and entries[0].id else 1,
            )

            # Find the entry
            entry = None
            for e in entries:
                if e.id == entry_id:
                    entry = e
                    break

            if not entry:
                # Try to get from database if not in current page
                all_entries = self.repository.get_all()
                for e in all_entries:
                    if e.id == entry_id:
                        entry = e
                        break

            if not entry:
                self.console.print(
                    f"❌ Entry with ID {entry_id} not found.", style="red"
                )
                return

            # Create detailed view
            details = Table(title=f"Knowledge Base Entry Details (ID: {entry.id})")
            details.add_column("Field", style="bold cyan")
            details.add_column("Value", style="white")

            details.add_row("ID", str(entry.id) if entry.id else "N/A")
            details.add_row("Run ID", str(entry.run_id))
            details.add_row("Task ID", str(entry.task_id))
            details.add_row("Setup ID", str(entry.setup_id))
            details.add_row("Flow ID", str(entry.flow_id))
            details.add_row("Algorithm (Flow)", entry.flow_name)
            details.add_row("Algorithm Family", entry.algo_family)
            details.add_row("Data ID", str(entry.data_id))
            details.add_row("Dataset", entry.data_name)

            # Display metrics
            if entry.metrics:
                for metric_name, metric_value in entry.metrics.items():
                    details.add_row(f"Metric: {metric_name}", f"{metric_value:.6f}")
            else:
                details.add_row("Metrics", "No metrics available")

            if entry.meta_vector:
                vector_info = f"[{len(entry.meta_vector)} dimensions]"
                if len(entry.meta_vector) <= 5:
                    vector_info += f" {entry.meta_vector}"
                else:
                    vector_info += (
                        f" [{entry.meta_vector[0]:.3f}, "
                        f"{entry.meta_vector[1]:.3f}, ..., "
                        f"{entry.meta_vector[-1]:.3f}]"
                    )
                details.add_row("Meta Vector", vector_info)
            else:
                details.add_row("Meta Vector", "Not computed")

            details.add_row(
                "Created At",
                entry.created_at.isoformat() if entry.created_at else "N/A",
            )
            details.add_row(
                "Updated At",
                entry.updated_at.isoformat() if entry.updated_at else "N/A",
            )

            self.console.print(details)

            Prompt.ask("Press Enter to continue", default="")

        except (ValueError, KeyError) as e:
            self.console.print(f"❌ Error showing entry details: {e}", style="red")

    def run(self) -> None:
        """Main application loop"""
        setup()
        self.show_welcome()

        while True:
            self.show_menu()

            try:
                choice = IntPrompt.ask(
                    "Select an option", choices=["1", "2", "3"], default=1
                )

                self.console.print()

                if choice == 1:
                    # Get current entry count
                    try:
                        current_count = self.repository.count_entries()
                    except (RuntimeError, ConnectionError):
                        current_count = 0

                    # Show warning and get confirmation
                    warning_text = Text()
                    warning_text.append(
                        "⚠️  WARNING: Full Knowledge Base Update", style="bold red"
                    )
                    warning_text.append("\n\nThis operation will:")
                    warning_text.append(
                        f"\n• Delete ALL {current_count} existing knowledge base entries"
                    )
                    warning_text.append("\n• Download fresh data from OpenML")
                    warning_text.append(
                        "\n• Rebuild the entire knowledge base from scratch"
                    )
                    warning_text.append(
                        "\n\nThis process may take several minutes and cannot be undone.",
                        style="yellow",
                    )

                    warning_panel = Panel(
                        warning_text,
                        title="⚠️  Destructive Operation Warning",
                        border_style="red",
                        padding=(1, 2),
                    )
                    self.console.print(warning_panel)

                    # Get confirmation
                    if Confirm.ask(
                        "\nDo you want to continue with the full knowledge base update?",
                        default=False,
                    ):
                        self.console.print(
                            "🔄 Updating knowledge base...", style="bold yellow"
                        )
                        self.knowledge_builder_service.update_knowledge_base()
                        self.console.print(
                            "✅ Knowledge base updated successfully!",
                            style="bold green",
                        )
                    else:
                        self.console.print(
                            "❌ Knowledge base update cancelled.", style="yellow"
                        )
                elif choice == 2:
                    self.console.print(
                        "📊 Inspecting knowledge base...", style="bold magenta"
                    )
                    self.inspect_knowledge_base()
                elif choice == 3:
                    self.console.print(
                        "👋 Goodbye! Thanks for using Meta-Learning MVP!",
                        style="bold blue",
                    )
                    break

            except KeyboardInterrupt:
                self.console.print(
                    "\n\n👋 Goodbye! Thanks for using Meta-Learning MVP!",
                    style="bold blue",
                )
                break
            except (RuntimeError, ConnectionError) as e:
                self.console.print(f"❌ Service error: {e}", style="red")


@click.command()
def main() -> None:
    """
    Meta-Learning MVP CLI Application

    A command-line tool for meta-learning with classifier ensembles
    using OpenML datasets with latent space representation.

    \b
    Interactive Mode (default):
        meta-learning-cli
    """

    cli = MetaLearningCLI()
    cli.run()


if __name__ == "__main__":
    main()
