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
    run_id: int
    task_id: int
    setup_id: int
    flow_id: int
    flow_name: str
    data_id: int
    data_name: str
    eval_metric: str
    eval_value: float
    meta_vector: Optional[List[float]] = None  # Latent space vector representation can be calculated later for performance reasons
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'run_id': self.run_id,
            'task_id': self.task_id,
            'setup_id': self.setup_id,
            'flow_id': self.flow_id,
            'flow_name': self.flow_name,
            'data_id': self.data_id,
            'data_name': self.data_name,
            'eval_metric': self.eval_metric,
            'eval_value': self.eval_value,
            'meta_vector': self.meta_vector,
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
                
                # Serialize meta_vector as JSON if it exists
                meta_vector_json = json.dumps(entry.meta_vector) if entry.meta_vector else None
                
                cursor.execute("""
                    INSERT INTO knowledge_base 
                    (run_id, task_id, setup_id, flow_id, flow_name, data_id, data_name, 
                     eval_metric, eval_value, meta_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (entry.run_id, entry.task_id, entry.setup_id, entry.flow_id, 
                      entry.flow_name, entry.data_id, entry.data_name, entry.eval_metric,
                      entry.eval_value, meta_vector_json))
                
                conn.commit()
                entry_id = cursor.lastrowid
                
                self.logger.info(
                    "Inserted knowledge base entry with ID %s (run_id: %s, flow: %s, eval_value: %s)", 
                    entry_id, entry.run_id, entry.flow_name, entry.eval_value
                )
                
                return entry_id
                
        except sqlite3.Error as e:
            self.logger.error("Failed to insert knowledge base entry: %s", e)
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
                    SELECT id, run_id, task_id, setup_id, flow_id, flow_name, 
                           data_id, data_name, eval_metric, eval_value, meta_vector,
                           created_at, updated_at
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