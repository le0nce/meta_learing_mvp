"""
Meta-Learning MVP CLI Application

A command-line interface for meta-learning with classifier ensembles using latent space.
Provides functionality to manage datasets, find best classifiers, and create ensembles using latent space.
"""

import logging
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.text import Text

from app.config.config import settings
from app.logger_config import setup_logger
from app.services.knowledge_builder_service import KnowledgeBuilderService
from app.database.models import KnowledgeBaseRepository


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
        welcome_text.append("A tool for meta-learning with classifier ensembles using OpenML datasets", style="italic")
        
        panel = Panel(
            welcome_text,
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def show_menu(self) -> None:
        """Display the main menu options"""
        # Get knowledge base stats
        stats = self.knowledge_builder_service.get_knowledge_base_stats()
        
        menu_text = Text()
        menu_text.append("Available Actions:\n\n", style="bold")
        menu_text.append("1. 🔄 Update Knowledge Base manually\n", style="cyan")
        menu_text.append("2. ❌ Exit\n", style="red")
        
        if stats:
            menu_text.append(f"\nCurrent Knowledge Base:\n", style="bold dim")
            menu_text.append(f"  • Total entries: {stats.get('total_entries', 0)}\n", style="dim")
            menu_text.append(f"  • Average accuracy: {stats.get('average_accuracy', 0):.3f}\n", style="dim")
        
        panel = Panel(
            menu_text,
            title="Main Menu",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)

    def run(self) -> None:
        """Main application loop"""
        setup()
        self.show_welcome()
        
        while True:
            self.show_menu()
            
            try:
                choice = IntPrompt.ask(
                    "Select an option",
                    choices=["1", "2"],
                    default=1
                )
                
                self.console.print()
                
                if choice == 1:
                    self.console.print("🔄 Updating knowledge base...", style="bold yellow")
                    self.knowledge_builder_service.update_knowledge_base()
                    self.console.print("✅ Knowledge base updated successfully!", style="bold green")
                elif choice == 2:
                    self.console.print("👋 Goodbye! Thanks for using Meta-Learning MVP!", style="bold blue")
                    break
                    
                    
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Goodbye! Thanks for using Meta-Learning MVP!", style="bold blue")
                break
            except Exception as e:
                self.console.print(f"❌ Unexpected error: {e}", style="red")


@click.command()
def main() -> None:
    """
    Meta-Learning MVP CLI Application
    
    A command-line tool for meta-learning with classifier ensembles using OpenML datasets with latent space representation.
    
    \b
    Interactive Mode (default):
        meta-learning-cli
    """
    
    cli = MetaLearningCLI()
    cli.run()

if __name__ == "__main__":
    main()
