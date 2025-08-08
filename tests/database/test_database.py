"""
Comprehensive tests for database functionality
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from app.database.connection import DatabaseConnection
from app.database.models import KnowledgeBaseEntry, KnowledgeBaseRepository


class TestKnowledgeBaseEntry:
    """Test KnowledgeBaseEntry data model"""

    def test_to_dict_complete_entry(self):
        """Test to_dict method with complete entry"""
        created_at = datetime.now()
        updated_at = datetime.now()

        entry = KnowledgeBaseEntry(
            id=1,
            run_id=123,
            task_id=456,
            setup_id=789,
            flow_id=101112,
            flow_name="test.classifier",
            algo_family="tree",
            data_id=131415,
            data_name="test_dataset",
            metrics={"accuracy": 0.95, "f1_score": 0.92},
            meta_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            created_at=created_at,
            updated_at=updated_at,
        )

        result = entry.to_dict()

        assert result["id"] == 1
        assert result["run_id"] == 123
        assert result["task_id"] == 456
        assert result["setup_id"] == 789
        assert result["flow_id"] == 101112
        assert result["flow_name"] == "test.classifier"
        assert result["algo_family"] == "tree"
        assert result["data_id"] == 131415
        assert result["data_name"] == "test_dataset"
        assert result["metrics"] == {"accuracy": 0.95, "f1_score": 0.92}
        assert result["meta_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result["created_at"] == created_at.isoformat()
        assert result["updated_at"] == updated_at.isoformat()

    def test_to_dict_minimal_entry(self):
        """Test to_dict method with minimal entry"""
        entry = KnowledgeBaseEntry(
            run_id=123,
            task_id=456,
            setup_id=789,
            flow_id=101112,
            flow_name="test.classifier",
            algo_family="tree",
            data_id=131415,
            data_name="test_dataset",
            metrics={"accuracy": 0.95},
        )

        result = entry.to_dict()

        assert result["id"] is None
        assert result["meta_vector"] is None
        assert result["created_at"] is None
        assert result["updated_at"] is None
        assert result["metrics"] == {"accuracy": 0.95}


class TestDatabaseConnection:
    """Test DatabaseConnection class"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            yield tmp.name
        # Cleanup
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    def test_init_with_custom_path(self, temp_db_path):
        """Test database initialization with custom path"""
        db = DatabaseConnection(temp_db_path)
        assert db.db_path == temp_db_path
        assert os.path.exists(temp_db_path)

    def test_init_with_default_path(self):
        """Test database initialization with default path"""
        with patch.object(Path, "mkdir") as mock_mkdir:
            db = DatabaseConnection()
            assert "knowledge_base.db" in db.db_path
            mock_mkdir.assert_called_once()

    def test_ensure_db_directory_creates_missing_directories(self):
        """Test that database directory is created if it doesn't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "path", "test.db")
            db = DatabaseConnection(nested_path)
            assert os.path.exists(os.path.dirname(nested_path))

    def test_health_check_success(self, temp_db_path):
        """Test successful health check"""
        db = DatabaseConnection(temp_db_path)
        assert db.health_check() is True

    def test_health_check_failure(self, temp_db_path):
        """Test health check failure with invalid database"""
        # Create a valid database first
        db = DatabaseConnection(temp_db_path)
        # Now override the path to an invalid one to simulate corruption
        db.db_path = "/nonexistent/invalid.db"
        assert db.health_check() is False

    def test_get_connection_context_manager(self, temp_db_path):
        """Test connection context manager"""
        db = DatabaseConnection(temp_db_path)
        with db.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            # Test that row_factory is set
            assert conn.row_factory == sqlite3.Row

    def test_database_tables_created(self, temp_db_path):
        """Test that required tables and indexes are created"""
        db = DatabaseConnection(temp_db_path)
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Check if knowledge_base table exists
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='knowledge_base'
            """
            )
            assert cursor.fetchone() is not None

            # Check if indexes exist
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='index' AND tbl_name='knowledge_base'
            """
            )
            indexes = cursor.fetchall()
            index_names = [idx[0] for idx in indexes]

            expected_indexes = [
                "idx_run_id",
                "idx_task_id",
                "idx_flow_id",
                "idx_data_id",
                "idx_algo_family",
            ]
            for expected in expected_indexes:
                assert expected in index_names

    def test_database_trigger_created(self, temp_db_path):
        """Test that update trigger is created"""
        db = DatabaseConnection(temp_db_path)
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name FROM sqlite_master 
                WHERE type='trigger' AND name='update_knowledge_base_timestamp'
            """
            )
            assert cursor.fetchone() is not None


class TestKnowledgeBaseRepository:
    """Test KnowledgeBaseRepository class"""

    @pytest.fixture
    def temp_db_setup(self):
        """Set up temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            temp_path = tmp.name

        db = DatabaseConnection(temp_path)
        repo = KnowledgeBaseRepository(db)

        yield repo, temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def sample_entry(self):
        """Create sample knowledge base entry"""
        return KnowledgeBaseEntry(
            run_id=12345,
            task_id=67890,
            setup_id=11111,
            flow_id=22222,
            flow_name="sklearn.ensemble.RandomForestClassifier",
            algo_family="ensemble",
            data_id=33333,
            data_name="iris_dataset",
            metrics={
                "predictive_accuracy": 0.96,
                "f1_score": 0.94,
                "precision": 0.95,
                "recall": 0.93,
            },
            meta_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )

    def test_repository_init_with_default_connection(self):
        """Test repository initialization with default connection"""
        repo = KnowledgeBaseRepository()
        assert repo.db is not None
        assert isinstance(repo.db, DatabaseConnection)

    def test_repository_init_with_custom_connection(self, temp_db_setup):
        """Test repository initialization with custom connection"""
        repo, _ = temp_db_setup
        assert repo.db is not None

    def test_insert_entry_success(self, temp_db_setup, sample_entry):
        """Test successful entry insertion"""
        repo, _ = temp_db_setup

        entry_id = repo.insert_entry(sample_entry)

        assert entry_id is not None
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_insert_entry_with_minimal_data(self, temp_db_setup):
        """Test insertion with minimal required data"""
        repo, _ = temp_db_setup

        minimal_entry = KnowledgeBaseEntry(
            run_id=1,
            task_id=1,
            setup_id=1,
            flow_id=1,
            flow_name="test",
            algo_family="test",
            data_id=1,
            data_name="test",
            metrics={},
        )

        entry_id = repo.insert_entry(minimal_entry)
        assert entry_id is not None

    def test_insert_entry_with_none_meta_vector(self, temp_db_setup):
        """Test insertion with None meta_vector"""
        repo, _ = temp_db_setup

        entry = KnowledgeBaseEntry(
            run_id=1,
            task_id=1,
            setup_id=1,
            flow_id=1,
            flow_name="test",
            algo_family="test",
            data_id=1,
            data_name="test",
            metrics={"accuracy": 0.8},
            meta_vector=None,
        )

        entry_id = repo.insert_entry(entry)
        assert entry_id is not None

    def test_insert_entry_database_error(self, temp_db_setup):
        """Test insertion failure due to database error"""
        repo, db_path = temp_db_setup

        # Corrupt the database by deleting it
        os.unlink(db_path)

        entry = KnowledgeBaseEntry(
            run_id=1,
            task_id=1,
            setup_id=1,
            flow_id=1,
            flow_name="test",
            algo_family="test",
            data_id=1,
            data_name="test",
            metrics={},
        )

        with pytest.raises(sqlite3.Error):
            repo.insert_entry(entry)

    def test_get_all_empty_database(self, temp_db_setup):
        """Test get_all with empty database"""
        repo, _ = temp_db_setup

        entries = repo.get_all()
        assert entries == []

    def test_get_all_single_entry(self, temp_db_setup, sample_entry):
        """Test get_all with single entry"""
        repo, _ = temp_db_setup

        entry_id = repo.insert_entry(sample_entry)
        entries = repo.get_all()

        assert len(entries) == 1
        retrieved = entries[0]
        assert retrieved.id == entry_id
        assert retrieved.run_id == sample_entry.run_id
        assert retrieved.flow_name == sample_entry.flow_name
        assert retrieved.metrics == sample_entry.metrics
        assert retrieved.meta_vector == sample_entry.meta_vector

    def test_get_all_multiple_entries(self, temp_db_setup):
        """Test get_all with multiple entries"""
        repo, _ = temp_db_setup

        # Insert multiple entries
        entries_data = []
        for i in range(5):
            entry = KnowledgeBaseEntry(
                run_id=1000 + i,
                task_id=2000 + i,
                setup_id=3000 + i,
                flow_id=4000 + i,
                flow_name=f"test_flow_{i}",
                algo_family=f"family_{i}",
                data_id=5000 + i,
                data_name=f"dataset_{i}",
                metrics={"accuracy": 0.8 + i * 0.02},
                meta_vector=[float(i)] * 3,
            )
            entry_id = repo.insert_entry(entry)
            entries_data.append((entry_id, entry))

        retrieved_entries = repo.get_all()
        assert len(retrieved_entries) == 5

        # Verify all entries were retrieved correctly
        retrieved_run_ids = {entry.run_id for entry in retrieved_entries}
        expected_run_ids = {1000 + i for i in range(5)}
        assert retrieved_run_ids == expected_run_ids

    def test_get_all_with_limit(self, temp_db_setup):
        """Test get_all with limit parameter"""
        repo, _ = temp_db_setup

        # Insert 5 entries
        for i in range(5):
            entry = KnowledgeBaseEntry(
                run_id=i,
                task_id=i,
                setup_id=i,
                flow_id=i,
                flow_name=f"flow_{i}",
                algo_family="test",
                data_id=i,
                data_name=f"data_{i}",
                metrics={"accuracy": 0.8},
            )
            repo.insert_entry(entry)

        # Get only 3 entries
        entries = repo.get_all(limit=3)
        assert len(entries) == 3

    def test_get_all_with_limit_and_offset(self, temp_db_setup):
        """Test get_all with limit and offset parameters"""
        repo, _ = temp_db_setup

        # Insert 10 entries
        inserted_run_ids = []
        for i in range(10):
            entry = KnowledgeBaseEntry(
                run_id=i,
                task_id=i,
                setup_id=i,
                flow_id=i,
                flow_name=f"flow_{i}",
                algo_family="test",
                data_id=i,
                data_name=f"data_{i}",
                metrics={"accuracy": 0.8},
            )
            repo.insert_entry(entry)
            inserted_run_ids.append(i)

        # Get 3 entries starting from offset 2
        entries = repo.get_all(limit=3, offset=2)
        assert len(entries) == 3

        # Verify we got 3 entries with different IDs
        retrieved_run_ids = {entry.run_id for entry in entries}
        assert len(retrieved_run_ids) == 3

        # Make sure they are all from our inserted entries
        assert retrieved_run_ids.issubset(set(inserted_run_ids))

    def test_count_entries_empty_database(self, temp_db_setup):
        """Test count_entries with empty database"""
        repo, _ = temp_db_setup

        count = repo.count_entries()
        assert count == 0

    def test_count_entries_with_data(self, temp_db_setup):
        """Test count_entries with data"""
        repo, _ = temp_db_setup

        # Insert 3 entries
        for i in range(3):
            entry = KnowledgeBaseEntry(
                run_id=i,
                task_id=i,
                setup_id=i,
                flow_id=i,
                flow_name=f"flow_{i}",
                algo_family="test",
                data_id=i,
                data_name=f"data_{i}",
                metrics={"accuracy": 0.8},
            )
            repo.insert_entry(entry)

        count = repo.count_entries()
        assert count == 3

    def test_clear_all_entries_empty_database(self, temp_db_setup):
        """Test clear_all_entries with empty database"""
        repo, _ = temp_db_setup

        deleted_count = repo.clear_all_entries()
        assert deleted_count == 0

    def test_clear_all_entries_with_data(self, temp_db_setup):
        """Test clear_all_entries with data"""
        repo, _ = temp_db_setup

        # Insert 5 entries
        for i in range(5):
            entry = KnowledgeBaseEntry(
                run_id=i,
                task_id=i,
                setup_id=i,
                flow_id=i,
                flow_name=f"flow_{i}",
                algo_family="test",
                data_id=i,
                data_name=f"data_{i}",
                metrics={"accuracy": 0.8},
            )
            repo.insert_entry(entry)

        # Verify entries exist
        assert repo.count_entries() == 5

        # Clear all entries
        deleted_count = repo.clear_all_entries()
        assert deleted_count == 5

        # Verify database is empty
        assert repo.count_entries() == 0
        assert repo.get_all() == []

    def test_row_to_entry_complete_data(self, temp_db_setup, sample_entry):
        """Test _row_to_entry with complete data"""
        repo, _ = temp_db_setup

        # Insert entry and retrieve it to test row conversion
        entry_id = repo.insert_entry(sample_entry)
        entries = repo.get_all()

        retrieved = entries[0]
        assert retrieved.id == entry_id
        assert retrieved.run_id == sample_entry.run_id
        assert retrieved.task_id == sample_entry.task_id
        assert retrieved.setup_id == sample_entry.setup_id
        assert retrieved.flow_id == sample_entry.flow_id
        assert retrieved.flow_name == sample_entry.flow_name
        assert retrieved.algo_family == sample_entry.algo_family
        assert retrieved.data_id == sample_entry.data_id
        assert retrieved.data_name == sample_entry.data_name
        assert retrieved.metrics == sample_entry.metrics
        assert retrieved.meta_vector == sample_entry.meta_vector
        assert retrieved.created_at is not None
        assert isinstance(retrieved.created_at, datetime)

    def test_row_to_entry_with_malformed_json(self, temp_db_setup):
        """Test _row_to_entry with malformed JSON data"""
        repo, db_path = temp_db_setup

        # Manually insert entry with malformed JSON
        with repo.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO knowledge_base 
                (run_id, task_id, setup_id, flow_id, flow_name, algo_family, 
                 data_id, data_name, metrics, meta_vector)
                VALUES (1, 1, 1, 1, 'test', 'test', 1, 'test', 
                        'invalid_json', 'invalid_json')
            """
            )
            conn.commit()

        entries = repo.get_all()
        assert len(entries) == 1

        # Should have empty/default values for malformed JSON
        retrieved = entries[0]
        assert retrieved.metrics == {}
        assert retrieved.meta_vector is None

    def test_row_to_entry_with_invalid_datetime(self, temp_db_setup):
        """Test _row_to_entry with invalid datetime strings"""
        repo, db_path = temp_db_setup

        # Manually insert entry with invalid datetime
        with repo.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO knowledge_base 
                (run_id, task_id, setup_id, flow_id, flow_name, algo_family, 
                 data_id, data_name, metrics, meta_vector, created_at, updated_at)
                VALUES (1, 1, 1, 1, 'test', 'test', 1, 'test', '{}', null,
                        'invalid_datetime', 'invalid_datetime')
            """
            )
            conn.commit()

        entries = repo.get_all()
        assert len(entries) == 1

        # Should have None for invalid datetime strings
        retrieved = entries[0]
        assert retrieved.created_at is None
        assert retrieved.updated_at is None

    def test_database_error_handling_in_operations(self, temp_db_setup):
        """Test error handling in various database operations"""
        repo, db_path = temp_db_setup

        # Close/corrupt the database
        os.unlink(db_path)

        # Test that all operations raise sqlite3.Error
        with pytest.raises(sqlite3.Error):
            repo.get_all()

        with pytest.raises(sqlite3.Error):
            repo.count_entries()

        with pytest.raises(sqlite3.Error):
            repo.clear_all_entries()


def test_database_integration():
    """Integration test for complete database workflow"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        test_db_path = tmp.name

    try:
        # Initialize database
        db = DatabaseConnection(test_db_path)
        assert db.health_check()

        repo = KnowledgeBaseRepository(db)

        # Test complete workflow
        entries_to_insert = []
        for i in range(10):
            entry = KnowledgeBaseEntry(
                run_id=1000 + i,
                task_id=2000 + i,
                setup_id=3000 + i,
                flow_id=4000 + i,
                flow_name=f"sklearn.ensemble.Method{i}",
                algo_family="ensemble" if i % 2 == 0 else "tree",
                data_id=5000 + i,
                data_name=f"dataset_{i}",
                metrics={
                    "accuracy": 0.85 + i * 0.01,
                    "f1_score": 0.80 + i * 0.01,
                    "precision": 0.88 + i * 0.01,
                },
                meta_vector=[float(i + j) for j in range(5)],
            )
            entries_to_insert.append(entry)

        # Insert all entries
        inserted_ids = []
        for entry in entries_to_insert:
            entry_id = repo.insert_entry(entry)
            inserted_ids.append(entry_id)

        # Verify count
        assert repo.count_entries() == 10

        # Test pagination
        page1 = repo.get_all(limit=5, offset=0)
        page2 = repo.get_all(limit=5, offset=5)

        assert len(page1) == 5
        assert len(page2) == 5

        # Verify no overlap between pages
        page1_ids = {entry.id for entry in page1}
        page2_ids = {entry.id for entry in page2}
        assert page1_ids.isdisjoint(page2_ids)

        # Test data integrity
        all_entries = repo.get_all()
        for retrieved_entry in all_entries:
            # Find corresponding original entry
            original_entry = None
            for original in entries_to_insert:
                if original.run_id == retrieved_entry.run_id:
                    original_entry = original
                    break

            assert original_entry is not None
            assert retrieved_entry.flow_name == original_entry.flow_name
            assert retrieved_entry.algo_family == original_entry.algo_family
            assert retrieved_entry.metrics == original_entry.metrics
            assert retrieved_entry.meta_vector == original_entry.meta_vector
            assert retrieved_entry.created_at is not None

        # Test clearing database
        deleted_count = repo.clear_all_entries()
        assert deleted_count == 10
        assert repo.count_entries() == 0
        assert repo.get_all() == []

    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)


if __name__ == "__main__":
    # Run basic test for backward compatibility
    test_database_integration()
    print("🎉 All comprehensive database tests completed!")
