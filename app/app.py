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
from rich.prompt import Confirm, IntPrompt
from rich.text import Text

from app.config.config import settings
from app.logger_config import setup_logger
from app.services import KnowledgeBuilderService
from app.services import OpenMLService


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
        # Dependency injection: inject console into service
        self.knowledge_builder = KnowledgeBuilderService(self.console)
        self.open_ml_service = OpenMLService(self.console)
    
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
        menu_text = Text()
        menu_text.append("Available Actions:\n\n", style="bold")
        menu_text.append("1. 📊 Build Meta-Knowledge-Base\n", style="cyan")
        menu_text.append("2. 🔧 Show OpenML Configuration\n", style="cyan")
        menu_text.append("3. ➕ Add Dataset from OpenML\n", style="green")
        menu_text.append("4. ➖ Delete Dataset from Configuration\n", style="yellow")
        menu_text.append("5. 📋 List All Configured Datasets\n", style="blue")
        menu_text.append("6. ❌ Exit\n", style="red")
        
        
        panel = Panel(
            menu_text,
            title="Main Menu",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def _handle_add_dataset(self) -> None:
        """Handle adding a dataset from OpenML"""
        try:
            self.console.print("➕ Add Dataset from OpenML", style="bold green")
            self.console.print("Enter the OpenML dataset ID you want to add to your configuration.", style="italic")
            self.console.print("You can find dataset IDs at: https://www.openml.org/search?type=data", style="dim")
            self.console.print()
            
            dataset_id = IntPrompt.ask("OpenML Dataset ID")
            
            if dataset_id <= 0:
                self.console.print("❌ Invalid dataset ID. Must be a positive integer.", style="bold red")
                return
            
            self.open_ml_service.add_dataset(dataset_id)
            
        except Exception as e:
            self.console.print(f"❌ Error: {e}", style="bold red")
    
    def _handle_delete_dataset(self) -> None:
        """Handle deleting a dataset from configuration"""
        try:
            self.console.print("➖ Delete Dataset from Configuration", style="bold yellow")
            self.console.print("Enter the dataset ID you want to remove from your configuration.", style="italic")
            self.console.print()
            
            # First show current datasets
            self.open_ml_service.list_datasets()
            self.console.print()
            
            if not Confirm.ask("Do you want to proceed with deleting a dataset?", default=True):
                return
            
            dataset_id = IntPrompt.ask("Dataset ID to delete")
            
            if dataset_id <= 0:
                self.console.print("❌ Invalid dataset ID. Must be a positive integer.", style="bold red")
                return
            
            # Confirm deletion
            if Confirm.ask(f"Are you sure you want to delete dataset {dataset_id} from configuration?", default=False):
                self.open_ml_service.delete_dataset(dataset_id)
            else:
                self.console.print("❌ Deletion cancelled.", style="yellow")
                
        except Exception as e:
            self.console.print(f"❌ Error: {e}", style="bold red")

    def run(self) -> None:
        """Main application loop"""
        setup()
        self.show_welcome()
        
        while True:
            self.show_menu()
            
            try:
                choice = IntPrompt.ask(
                    "Select an option",
                    choices=["1", "2", "3", "4", "5", "6"],
                    default=1
                )
                
                self.console.print()
                
                if choice == 1:
                    self.knowledge_builder.build_meta_knowledge_base()
                elif choice == 2:
                    self.open_ml_service.show_open_ml_config()
                elif choice == 3:
                    self._handle_add_dataset()
                elif choice == 4:
                    self._handle_delete_dataset()
                elif choice == 5:
                    self.open_ml_service.list_datasets()
                elif choice == 6:
                    self.console.print("👋 Goodbye! Thanks for using Meta-Learning MVP!", style="bold blue")
                    break
                
                # Ask if user wants to continue
                if choice != "exit":
                    if not Confirm.ask("\nWould you like to perform another action?", default=True):
                        self.console.print("👋 Goodbye! Thanks for using Meta-Learning MVP!", style="bold blue")
                        break
                    self.console.clear()
                    self.show_welcome()
                    
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
