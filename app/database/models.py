"""
Data models for the knowledge base
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import sqlite3

from app.database.connection import DatabaseConnection


@dataclass
class KnowledgeBaseEntry:
    """Data model for knowledge base entries"""
    meta_vector: List[float]
    open_ml_run_id: int
    open_ml_flow_name: str
    accuracy: float
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'meta_vector': self.meta_vector,
            'open_ml_run_id': self.open_ml_run_id,
            'open_ml_flow_name': self.open_ml_flow_name,
            'accuracy': self.accuracy,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class KnowledgeBaseRepository:
    """Repository for knowledge base operations"""
    
    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """
        Initialize repository
        
        Args:
            db_connection: Database connection instance
        """
        self.logger = logging.getLogger(__name__)
        self.db = db_connection or DatabaseConnection()
    
    def insert_entry(self, entry: KnowledgeBaseEntry) -> int:
        """
        Insert a new knowledge base entry
        
        Args:
            entry: Knowledge base entry to insert
            
        Returns:
            int: ID of the inserted entry
            
        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize meta_vector as JSON
                meta_vector_json = json.dumps(entry.meta_vector)
                
                cursor.execute("""
                    INSERT INTO knowledge_base 
                    (meta_vector, open_ml_run_id, open_ml_flow_name, accuracy)
                    VALUES (?, ?, ?, ?)
                """, (meta_vector_json, entry.open_ml_run_id, entry.open_ml_flow_name, entry.accuracy))
                
                conn.commit()
                entry_id = cursor.lastrowid
                
                self.logger.info(
                    "Inserted knowledge base entry with ID %s (run_id: %s, flow: %s)", 
                    entry_id, entry.open_ml_run_id, entry.open_ml_flow_name
                )
                
                return entry_id
                
        except sqlite3.Error as e:
            self.logger.error("Failed to insert knowledge base entry: %s", e)
            raise
    
    def get_by_id(self, entry_id: int) -> Optional[KnowledgeBaseEntry]:
        """
        Get knowledge base entry by ID
        
        Args:
            entry_id: ID of the entry to retrieve
            
        Returns:
            KnowledgeBaseEntry if found, None otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, meta_vector, open_ml_run_id, open_ml_flow_name, 
                           accuracy, created_at, updated_at
                    FROM knowledge_base 
                    WHERE id = ?
                """, (entry_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
                
        except sqlite3.Error as e:
            self.logger.error("Failed to get knowledge base entry by ID %s: %s", entry_id, e)
            raise
    
    def get_by_run_id(self, run_id: int) -> Optional[KnowledgeBaseEntry]:
        """
        Get knowledge base entry by OpenML run ID
        
        Args:
            run_id: OpenML run ID
            
        Returns:
            KnowledgeBaseEntry if found, None otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, meta_vector, open_ml_run_id, open_ml_flow_name, 
                           accuracy, created_at, updated_at
                    FROM knowledge_base 
                    WHERE open_ml_run_id = ?
                """, (run_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_entry(row)
                return None
                
        except sqlite3.Error as e:
            self.logger.error("Failed to get knowledge base entry by run ID %s: %s", run_id, e)
            raise
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[KnowledgeBaseEntry]:
        """
        Get all knowledge base entries
        
        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of knowledge base entries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, meta_vector, open_ml_run_id, open_ml_flow_name, 
                           accuracy, created_at, updated_at
                    FROM knowledge_base 
                    ORDER BY created_at DESC
                """
                
                params = []
                if limit is not None:
                    query += " LIMIT ? OFFSET ?"
                    params = [limit, offset]
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_entry(row) for row in rows]
                
        except sqlite3.Error as e:
            self.logger.error("Failed to get knowledge base entries: %s", e)
            raise
    
    def get_by_accuracy_range(self, min_accuracy: float, max_accuracy: float) -> List[KnowledgeBaseEntry]:
        """
        Get entries within accuracy range
        
        Args:
            min_accuracy: Minimum accuracy threshold
            max_accuracy: Maximum accuracy threshold
            
        Returns:
            List of knowledge base entries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, meta_vector, open_ml_run_id, open_ml_flow_name, 
                           accuracy, created_at, updated_at
                    FROM knowledge_base 
                    WHERE accuracy BETWEEN ? AND ?
                    ORDER BY accuracy DESC
                """, (min_accuracy, max_accuracy))
                
                rows = cursor.fetchall()
                return [self._row_to_entry(row) for row in rows]
                
        except sqlite3.Error as e:
            self.logger.error("Failed to get entries by accuracy range: %s", e)
            raise
    
    def count_entries(self) -> int:
        """
        Get total count of knowledge base entries
        
        Returns:
            int: Total number of entries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM knowledge_base")
                return cursor.fetchone()[0]
                
        except sqlite3.Error as e:
            self.logger.error("Failed to count knowledge base entries: %s", e)
            raise
    
    def delete_by_id(self, entry_id: int) -> bool:
        """
        Delete knowledge base entry by ID
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            bool: True if entry was deleted, False if not found
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM knowledge_base WHERE id = ?", (entry_id,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.info("Deleted knowledge base entry with ID %s", entry_id)
                else:
                    self.logger.warning("No knowledge base entry found with ID %s", entry_id)
                    
                return deleted
                
        except sqlite3.Error as e:
            self.logger.error("Failed to delete knowledge base entry: %s", e)
            raise
    
    def _row_to_entry(self, row) -> KnowledgeBaseEntry:
        """
        Convert database row to KnowledgeBaseEntry
        
        Args:
            row: Database row
            
        Returns:
            KnowledgeBaseEntry instance
        """
        # Parse meta_vector from JSON
        meta_vector = json.loads(row['meta_vector'])
        
        # Parse timestamps
        created_at = datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        updated_at = datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        
        return KnowledgeBaseEntry(
            id=row['id'],
            meta_vector=meta_vector,
            open_ml_run_id=row['open_ml_run_id'],
            open_ml_flow_name=row['open_ml_flow_name'],
            accuracy=row['accuracy'],
            created_at=created_at,
            updated_at=updated_at
        )
