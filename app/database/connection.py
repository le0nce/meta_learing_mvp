"""
Database connection and configuration for Meta-Learning MVP
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
import json
from contextlib import contextmanager

from app.config.config import settings


class DatabaseConnection:
    """Manages SQLite database connection and operations"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file. If None, uses default from settings.
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path or self._get_default_db_path()
        self._ensure_db_directory()
        self._init_database()
    
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        # Use project root directory for database storage
        project_root = Path(__file__).parent.parent.parent
        db_dir = project_root / "data"
        return str(db_dir / "knowledge_base.db")
    
    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists"""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self) -> None:
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create knowledge_base table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id INTEGER NOT NULL,
                        task_id INTEGER NOT NULL,
                        setup_id INTEGER NOT NULL,
                        flow_id INTEGER NOT NULL,
                        flow_name TEXT NOT NULL,
                        algo_family TEXT NOT NULL,
                        data_id INTEGER NOT NULL,
                        data_name TEXT NOT NULL,
                        metrics TEXT NOT NULL,  -- JSON serialized metrics dict
                        meta_vector TEXT,  -- JSON serialized vector, optional
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_run_id 
                    ON knowledge_base(run_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_task_id 
                    ON knowledge_base(task_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_flow_id 
                    ON knowledge_base(flow_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_data_id 
                    ON knowledge_base(data_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_algo_family 
                    ON knowledge_base(algo_family)
                """)
                
                # Create trigger to update updated_at timestamp
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_knowledge_base_timestamp 
                    AFTER UPDATE ON knowledge_base
                    FOR EACH ROW
                    BEGIN
                        UPDATE knowledge_base 
                        SET updated_at = CURRENT_TIMESTAMP 
                        WHERE id = NEW.id;
                    END
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully at: %s", self.db_path)
                
        except sqlite3.Error as e:
            self.logger.error("Failed to initialize database: %s", e)
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error("Database error: %s", e)
            raise
        finally:
            if conn:
                conn.close()
    
    def health_check(self) -> bool:
        """
        Check if database is accessible and healthy
        
        Returns:
            bool: True if database is healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return True
        except sqlite3.Error as e:
            self.logger.error("Database health check failed: %s", e)
            return False
