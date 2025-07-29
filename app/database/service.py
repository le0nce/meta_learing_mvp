"""
Database service for managing knowledge base operations
"""

import logging
from typing import List, Optional

from app.database.connection import DatabaseConnection
from app.database.models import KnowledgeBaseRepository, KnowledgeBaseEntry


class DatabaseService:
    """High-level service for database operations"""
    
    def __init__(self):
        """Initialize database service"""
        self.logger = logging.getLogger(__name__)
        self.connection = DatabaseConnection()
        self.repository = KnowledgeBaseRepository(self.connection)
    
    def initialize(self) -> bool:
        """
        Initialize and verify database connection
        
        Returns:
            bool: True if initialization successful
        """
        try:
            health_ok = self.connection.health_check()
            if health_ok:
                self.logger.info("Database service initialized successfully")
            else:
                self.logger.error("Database health check failed")
            return health_ok
        except Exception as e:
            self.logger.error("Failed to initialize database service: %s", e)
            return False
    
    def add_knowledge_entry(self, meta_vector: List[float], run_id: int, 
                          flow_name: str, accuracy: float) -> Optional[int]:
        """
        Add a new knowledge base entry
        
        Args:
            meta_vector: Vector representation in latent space
            run_id: OpenML run ID
            flow_name: OpenML flow name
            accuracy: Model accuracy
            
        Returns:
            Entry ID if successful, None otherwise
        """
        try:
            entry = KnowledgeBaseEntry(
                meta_vector=meta_vector,
                open_ml_run_id=run_id,
                open_ml_flow_name=flow_name,
                accuracy=accuracy
            )
            return self.repository.insert_entry(entry)
        except Exception as e:
            self.logger.error("Failed to add knowledge entry: %s", e)
            return None
    
    def get_best_performers(self, limit: int = 10) -> List[KnowledgeBaseEntry]:
        """
        Get the best performing entries by accuracy
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of top performing entries
        """
        try:
            entries = self.repository.get_all(limit=limit)
            return sorted(entries, key=lambda x: x.accuracy, reverse=True)[:limit]
        except Exception as e:
            self.logger.error("Failed to get best performers: %s", e)
            return []
    
    def search_by_accuracy(self, min_accuracy: float) -> List[KnowledgeBaseEntry]:
        """
        Search entries by minimum accuracy threshold
        
        Args:
            min_accuracy: Minimum accuracy threshold
            
        Returns:
            List of entries meeting the criteria
        """
        try:
            return self.repository.get_by_accuracy_range(min_accuracy, 1.0)
        except Exception as e:
            self.logger.error("Failed to search by accuracy: %s", e)
            return []
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        try:
            return {
                'total_entries': self.repository.count_entries(),
                'database_path': self.connection.db_path,
                'is_healthy': self.connection.health_check()
            }
        except Exception as e:
            self.logger.error("Failed to get database stats: %s", e)
            return {}
