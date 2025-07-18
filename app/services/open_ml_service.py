"""
OpenML Service
This service handles all operations regarding OpenML datasets
and configurations used in building the meta-knowledge base.
"""

import json
import logging
import openml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from app.services.console_protocol import ConsoleProtocol

class OpenMLService:
    """Service for building meta-knowledge base"""
    
    def __init__(self, console: ConsoleProtocol):
        """
        Initialize the knowledge builder service
        
        Args:
            console: Console instance for output (dependency injection)
        """
        self.console = console
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(__file__).parent.parent / "config" / "openMLConfig.json"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Error reading configuration: {e}")
            return {}

    def _save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to JSON file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            self.console.print(f"❌ Error saving configuration: {e}", style="bold red")
            return False

3    def add_dataset(self, dataset_id: int) -> None:
        """
        Add a dataset to the configuration by fetching its information from OpenML API
        and automatically analyzing its structure to create column specifications
        
        Args:
            dataset_id: The OpenML dataset ID to add
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Fetching dataset {dataset_id} from OpenML...", total=None)
                
                # Fetch dataset from OpenML API
                dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
                
                progress.update(task, description=f"Analyzing dataset structure...")
                
                # Load current configuration
                config = self._load_config()
                
                # Check if dataset already exists
                if dataset.name in config:
                    self.console.print(f"⚠️ Dataset '{dataset.name}' already exists in configuration!", style="bold yellow")
                    return
                
                # Get the data to analyze column types
                X, y, categorical_indicator, feature_names = dataset.get_data(
                    dataset_format="dataframe", target=dataset.default_target_attribute
                )
                
                # Combine features and target for analysis
                df = X.copy()
                if y is not None:
                    df[dataset.default_target_attribute] = y
                
                progress.update(task, description=f"Classifying column types...")
                
                # Analyze and classify columns
                dataset_config = self._analyze_dataset_structure(
                    df, dataset.default_target_attribute, categorical_indicator, feature_names
                )
                
                # Add to configuration using dataset name as key
                config[dataset.name] = dataset_config
                
                # Save configuration
                if self._save_config(config):
                    progress.update(task, description=f"✅ Dataset {dataset_id} added successfully!")
                    self.console.print(f"✅ Dataset '{dataset.name}' (ID: {dataset_id}) added to configuration!", style="bold green")
                    
                    # Show brief info about added dataset
                    info_panel = Panel(
                        f"[bold]Name:[/bold] {dataset.name}\n"
                        f"[bold]Target:[/bold] {dataset_config['target']}\n"
                        f"[bold]Categorical Columns:[/bold] {len(dataset_config.get('cat_cols', []))}\n"
                        f"[bold]Date Columns:[/bold] {len(dataset_config.get('date_cols', []))}\n"
                        f"[bold]Float Columns:[/bold] {len(dataset_config.get('float_cols', []))}",
                        title=f"📊 Dataset {dataset_id} Auto-Analyzed",
                        border_style="green"
                    )
                    self.console.print(info_panel)
        
        except openml.exceptions.OpenMLServerException as e:
            self.logger.error(f"OpenML API error for dataset {dataset_id}: {e}")
            self.console.print(f"❌ Dataset {dataset_id} not found on OpenML or API error: {e}", style="bold red")
        except Exception as e:
            self.logger.error(f"Error adding dataset {dataset_id}: {e}")
            self.console.print(f"❌ Error adding dataset {dataset_id}: {e}", style="bold red")

    def _analyze_dataset_structure(self, df: pd.DataFrame, target_col: str, 
                                 categorical_indicator: List[bool], 
                                 feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze dataset structure and automatically classify column types
        
        Args:
            df: The dataset dataframe
            target_col: Name of the target column
            categorical_indicator: Boolean list indicating categorical features
            feature_names: List of feature names
            
        Returns:
            Dictionary with classified column types
        """
        config = {
            "target": target_col,
            "id_cols": [],
            "cat_cols": [],
            "date_cols": [],
            "float_cols": [],
            "allow_negatives": []
        }
        
        for i, (col_name, dtype) in enumerate(df.dtypes.items()):
            if col_name == target_col:
                continue
                
            # Check if it's an ID column (unique values, often integer)
            if self._is_id_column(df[col_name]):
                config["id_cols"].append(col_name)
            
            # Check if it's a date column
            elif self._is_date_column(df[col_name], col_name):
                config["date_cols"].append(col_name)
            
            # Check if it's categorical (using OpenML's indicator or data analysis)
            elif (i < len(categorical_indicator) and categorical_indicator[i]) or self._is_categorical_column(df[col_name]):
                config["cat_cols"].append(col_name)
            
            # Check if it's a float/numeric column
            elif pd.api.types.is_numeric_dtype(dtype):
                config["float_cols"].append(col_name)
                
                # Check if column can have negative values
                if df[col_name].min() < 0:
                    config["allow_negatives"].append(col_name)
        
        return config
    
    def _is_id_column(self, series: pd.Series) -> bool:
        """Check if a column is likely an ID column"""
        # ID columns typically have unique values and are often integers
        unique_ratio = series.nunique() / len(series)
        is_mostly_unique = unique_ratio > 0.95
        
        # Check for common ID column patterns
        col_name_lower = series.name.lower() if series.name else ""
        has_id_name = any(keyword in col_name_lower for keyword in ['id', 'index', 'key', 'code'])
        
        return is_mostly_unique and (has_id_name or pd.api.types.is_integer_dtype(series))
    
    def _is_date_column(self, series: pd.Series, col_name: str) -> bool:
        """Check if a column is a date column"""
        # Check column name for date indicators
        col_name_lower = col_name.lower() if col_name else ""
        has_date_name = any(keyword in col_name_lower for keyword in ['date', 'time', 'day', 'month', 'year'])
        
        # Check if data type is datetime-related
        is_datetime_type = pd.api.types.is_datetime64_any_dtype(series)
        
        # Try to parse a sample of values as dates
        could_be_date = False
        if not is_datetime_type and has_date_name:
            try:
                sample = series.dropna().head(10)
                pd.to_datetime(sample, errors='raise')
                could_be_date = True
            except:
                could_be_date = False
        
        return is_datetime_type or could_be_date
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if a column should be treated as categorical"""
        # String columns are usually categorical
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            return True
        
        # Numeric columns with few unique values might be categorical
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.05 and series.nunique() < 20
        
        return False

    def delete_dataset(self, dataset_name: str) -> None:
        """
        Delete a dataset from the configuration by its name
        
        Args:
            dataset_name: The dataset name to remove
        """
        try:
            # Load current configuration
            config = self._load_config()
            
            # Check if dataset exists and remove it
            if dataset_name in config:
                del config[dataset_name]
                
                if self._save_config(config):
                    self.console.print(f"✅ Dataset '{dataset_name}' removed from configuration!", style="bold green")
                else:
                    self.console.print(f"❌ Failed to save configuration after removing dataset '{dataset_name}'", style="bold red")
            else:
                self.console.print(f"⚠️ Dataset '{dataset_name}' not found in configuration!", style="bold yellow")
        
        except Exception as e:
            self.logger.error(f"Error deleting dataset {dataset_name}: {e}")
            self.console.print(f"❌ Error deleting dataset {dataset_name}: {e}", style="bold red")

    def list_datasets(self) -> None:
        """List all datasets in the configuration with basic information"""
        try:
            config = self._load_config()
            
            if not config:
                self.console.print("📋 No datasets configured yet.", style="bold yellow")
                return
            
            datasets_table = Table(title="📊 Configured Datasets", show_header=True, header_style="bold cyan")
            datasets_table.add_column("Name", style="green", width=20)
            datasets_table.add_column("Target", style="blue", width=20)
            datasets_table.add_column("ID Cols", style="yellow", width=10)
            datasets_table.add_column("Cat Cols", style="magenta", width=10)
            datasets_table.add_column("Date Cols", style="cyan", width=10)
            datasets_table.add_column("Float Cols", style="white", width=12)
            datasets_table.add_column("Allow Negatives", style="red", width=15)
            
            for dataset_name, dataset_config in config.items():
                datasets_table.add_row(
                    dataset_name,
                    dataset_config.get('target', 'N/A'),
                    str(len(dataset_config.get('id_cols', []))),
                    str(len(dataset_config.get('cat_cols', []))),
                    str(len(dataset_config.get('date_cols', []))),
                    str(len(dataset_config.get('float_cols', []))),
                    str(len(dataset_config.get('allow_negatives', [])))
                )
            
            self.console.print(datasets_table)
            self.console.print(f"\n📈 Total datasets: {len(config)}")
            
        except Exception as e:
            self.logger.error(f"Error listing datasets: {e}")
            self.console.print(f"❌ Error listing datasets: {e}", style="bold red")

    def show_open_ml_config(self) -> None:
        """
        Show OpenML configuration.

        OpenML configuration holds definition of datasets with their column specifications
        that are used to build meta-knowledge base.
        """
        
        try:
            # Load configuration from JSON file
            config = self._load_config()
            
            # Display configuration overview
            self.console.print("🔧 OpenML Configuration:", style="bold green")
            self.console.print("The datasets used for building the meta-knowledge base are defined in this configuration file.", style="italic")
            self.console.print("Each dataset includes column type specifications (ID, categorical, date, float, etc.).", style="italic")
            self.console.print()
            
            # Display datasets
            if config:
                # Use the list_datasets method for consistency
                self.list_datasets()
            else:
                self.console.print("📋 No datasets configured yet. Use 'add-dataset' command to add dataset configurations.", style="bold yellow")
            
        except Exception as e:
            self.logger.error(f"Error reading OpenML configuration: {e}")
            self.console.print(f"❌ Error reading configuration: {e}", style="bold red")

    def get_dataset_config(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a specific dataset
        
        Args:
            dataset_name: The name of the dataset
            
        Returns:
            Dictionary with dataset configuration or None if not found
        """
        try:
            config = self._load_config()
            return config.get(dataset_name)
        except Exception as e:
            self.logger.error(f"Error getting dataset config for {dataset_name}: {e}")
            return None

    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists in the configuration
        
        Args:
            dataset_name: The name of the dataset to check
            
        Returns:
            True if dataset exists, False otherwise
        """
        try:
            config = self._load_config()
            return dataset_name in config
        except Exception as e:
            self.logger.error(f"Error checking dataset existence for {dataset_name}: {e}")
            return False

    def get_all_dataset_names(self) -> list[str]:
        """
        Get all dataset names from the configuration
        
        Returns:
            List of dataset names
        """
        try:
            config = self._load_config()
            return list(config.keys())
        except Exception as e:
            self.logger.error(f"Error getting dataset names: {e}")
            return []

    def show_dataset_details(self, dataset_name: str) -> None:
        """
        Show detailed information about a specific dataset
        
        Args:
            dataset_name: The name of the dataset to show details for
        """
        try:
            dataset_config = self.get_dataset_config(dataset_name)
            
            if not dataset_config:
                self.console.print(f"⚠️ Dataset '{dataset_name}' not found in configuration!", style="bold yellow")
                return
            
            # Create detailed info panel
            details = []
            details.append(f"[bold]Target Column:[/bold] {dataset_config.get('target', 'N/A')}")
            
            id_cols = dataset_config.get('id_cols', [])
            if id_cols:
                details.append(f"[bold]ID Columns:[/bold] {', '.join(id_cols)}")
            
            cat_cols = dataset_config.get('cat_cols', [])
            if cat_cols:
                details.append(f"[bold]Categorical Columns:[/bold] {', '.join(cat_cols)}")
            
            date_cols = dataset_config.get('date_cols', [])
            if date_cols:
                details.append(f"[bold]Date Columns:[/bold] {', '.join(date_cols)}")
            
            float_cols = dataset_config.get('float_cols', [])
            if float_cols:
                details.append(f"[bold]Float Columns:[/bold] {', '.join(float_cols)}")
            
            allow_negatives = dataset_config.get('allow_negatives', [])
            if allow_negatives:
                details.append(f"[bold]Allow Negatives:[/bold] {', '.join(allow_negatives)}")
            
            info_panel = Panel(
                "\n".join(details),
                title=f"📊 Dataset '{dataset_name}' Configuration",
                border_style="blue"
            )
            self.console.print(info_panel)
            
        except Exception as e:
            self.logger.error(f"Error showing dataset details for {dataset_name}: {e}")
            self.console.print(f"❌ Error showing dataset details: {e}", style="bold red")

    def add_dataset_manual(self, dataset_name: str, dataset_config: Dict[str, Any]) -> None:
        """
        Manually add a dataset configuration to the OpenML config
        
        Args:
            dataset_name: The name/key of the dataset
            dataset_config: Dictionary containing the dataset configuration
        """
        try:
            # Load current configuration
            config = self._load_config()
            
            # Check if dataset already exists
            if dataset_name in config:
                self.console.print(f"⚠️ Dataset '{dataset_name}' already exists in configuration!", style="bold yellow")
                return
            
            # Add the dataset configuration
            config[dataset_name] = dataset_config
            
            # Save configuration
            if self._save_config(config):
                self.console.print(f"✅ Dataset '{dataset_name}' added to configuration!", style="bold green")
                
                # Show brief info about added dataset
                info_panel = Panel(
                    f"[bold]Name:[/bold] {dataset_name}\n"
                    f"[bold]Target:[/bold] {dataset_config.get('target', 'N/A')}\n"
                    f"[bold]Categorical Columns:[/bold] {len(dataset_config.get('cat_cols', []))}\n"
                    f"[bold]Date Columns:[/bold] {len(dataset_config.get('date_cols', []))}\n"
                    f"[bold]Float Columns:[/bold] {len(dataset_config.get('float_cols', []))}",
                    title=f"📊 Dataset {dataset_name} Config",
                    border_style="green"
                )
                self.console.print(info_panel)
        
        except Exception as e:
            self.logger.error(f"Error adding dataset {dataset_name}: {e}")
            self.console.print(f"❌ Error adding dataset {dataset_name}: {e}", style="bold red")

