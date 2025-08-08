"""
Data models for the knowledge base
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.database.connection import DatabaseConnection


@dataclass
class KnowledgeBaseEntry:  # pylint: disable=too-many-instance-attributes
    """Data model for knowledge base entries"""

    run_id: int
    task_id: int
    setup_id: int
    flow_id: int
    flow_name: str
    algo_family: str
    data_id: int
    data_name: str
    metrics: dict
    meta_vector: Optional[List[float]] = (
        None  # Latent space vector representation can be calculated later for performance reasons
    )
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "setup_id": self.setup_id,
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "algo_family": self.algo_family,
            "data_id": self.data_id,
            "data_name": self.data_name,
            "metrics": self.metrics,
            "meta_vector": self.meta_vector,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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
                meta_vector_json = (
                    json.dumps(entry.meta_vector) if entry.meta_vector else None
                )
                # Serialize metrics as JSON
                metrics_json = json.dumps(entry.metrics) if entry.metrics else "{}"

                cursor.execute(
                    """
                    INSERT INTO knowledge_base 
                    (run_id, task_id, setup_id, flow_id, flow_name, algo_family, data_id, data_name, 
                     metrics, meta_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.run_id,
                        entry.task_id,
                        entry.setup_id,
                        entry.flow_id,
                        entry.flow_name,
                        entry.algo_family,
                        entry.data_id,
                        entry.data_name,
                        metrics_json,
                        meta_vector_json,
                    ),
                )

                conn.commit()
                entry_id = cursor.lastrowid

                self.logger.info(
                    "Inserted knowledge base entry with ID %s "
                    "(run_id: %s, flow: %s, algo_family: %s)",
                    entry_id,
                    entry.run_id,
                    entry.flow_name,
                    entry.algo_family,
                )

                return entry_id

        except sqlite3.Error as e:
            self.logger.error("Failed to insert knowledge base entry: %s", e)
            raise

    def get_all(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[KnowledgeBaseEntry]:
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
                    SELECT id, run_id, task_id, setup_id, flow_id, flow_name, algo_family,
                           data_id, data_name, metrics, meta_vector,
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

    def count_entries(self) -> int:
        """
        Get the total count of knowledge base entries

        Returns:
            int: Total number of entries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM knowledge_base")
                count = cursor.fetchone()[0]
                return count

        except sqlite3.Error as e:
            self.logger.error("Failed to count knowledge base entries: %s", e)
            raise

    def clear_all_entries(self) -> int:
        """
        Clear all knowledge base entries

        Returns:
            int: Number of entries deleted

        Raises:
            sqlite3.Error: If database operation fails
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # First count the entries that will be deleted
                cursor.execute("SELECT COUNT(*) FROM knowledge_base")
                count_before = cursor.fetchone()[0]

                # Delete all entries
                cursor.execute("DELETE FROM knowledge_base")
                conn.commit()

                self.logger.info("Cleared %s entries from knowledge base", count_before)
                return count_before

        except sqlite3.Error as e:
            self.logger.error("Failed to clear knowledge base entries: %s", e)
            raise

    def _row_to_entry(self, row: tuple) -> KnowledgeBaseEntry:
        """
        Convert a database row to a KnowledgeBaseEntry object

        Args:
            row: Database row tuple

        Returns:
            KnowledgeBaseEntry object
        """
        # Parse created_at and updated_at from ISO format strings if they exist
        created_at = None
        updated_at = None

        if row[11]:  # created_at
            try:
                created_at = datetime.fromisoformat(row[11])
            except ValueError:
                self.logger.warning("Failed to parse created_at: %s", row[11])

        if row[12]:  # updated_at
            try:
                updated_at = datetime.fromisoformat(row[12])
            except ValueError:
                self.logger.warning("Failed to parse updated_at: %s", row[12])

        # Parse meta_vector from JSON if it exists
        meta_vector = None
        if row[10]:  # meta_vector
            try:
                meta_vector = json.loads(row[10])
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse meta_vector JSON: %s", row[10])

        # Parse metrics from JSON
        metrics = {}
        if row[9]:  # metrics
            try:
                metrics = json.loads(row[9])
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse metrics JSON: %s", row[9])

        return KnowledgeBaseEntry(
            id=row[0],
            run_id=row[1],
            task_id=row[2],
            setup_id=row[3],
            flow_id=row[4],
            flow_name=row[5],
            algo_family=row[6],
            data_id=row[7],
            data_name=row[8],
            metrics=metrics,
            meta_vector=meta_vector,
            created_at=created_at,
            updated_at=updated_at,
        )
