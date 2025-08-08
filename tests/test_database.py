"""
Test database functionality
"""

import os
import tempfile

from app.database.connection import DatabaseConnection
from app.database.models import KnowledgeBaseEntry, KnowledgeBaseRepository


def test_database():
    """Test basic database functionality"""
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        test_db_path = tmp.name

    try:
        # Test database connection
        print("🔧 Testing database connection...")
        db = DatabaseConnection(test_db_path)
        assert db.health_check(), "Database health check failed"
        print("✅ Database connection successful")

        # Test repository operations
        print("\n📝 Testing repository operations...")
        repo = KnowledgeBaseRepository(db)

        # Create test entry
        test_entry = KnowledgeBaseEntry(
            run_id=9999,
            task_id=1,
            setup_id=1,
            flow_id=1,
            flow_name="test.classifier",
            algo_family="tree",
            data_id=1,
            data_name="test_dataset",
            metrics={"predictive_accuracy": 0.95},
            meta_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        # Insert entry
        entry_id = repo.insert_entry(test_entry)
        assert entry_id is not None, "Failed to insert entry"
        print(f"✅ Entry inserted with ID: {entry_id}")

        # Retrieve entry - use get_all since get_by_id doesn't exist
        all_entries = repo.get_all()
        retrieved_entry = None
        for entry in all_entries:
            if entry.id == entry_id:
                retrieved_entry = entry
                break

        assert retrieved_entry is not None, "Failed to retrieve entry"
        assert retrieved_entry.run_id == 9999, "Run ID mismatch"
        assert retrieved_entry.metrics.get("predictive_accuracy") == 0.95, "Accuracy mismatch"
        print("✅ Entry retrieved successfully")

        # Test count
        count = repo.count_entries()
        assert count == 1, f"Expected 1 entry, got {count}"
        print(f"✅ Entry count correct: {count}")

        print("\n🎉 All database tests passed!")

    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
        print("🧹 Test database cleaned up")


if __name__ == "__main__":
    test_database()
