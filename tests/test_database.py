"""
Test database functionality
"""

import tempfile
import os
from pathlib import Path

from app.database.connection import DatabaseConnection
from app.database.models import KnowledgeBaseRepository, KnowledgeBaseEntry
from app.database.service import DatabaseService


def test_database():
    """Test basic database functionality"""
    # Create temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
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
            meta_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            open_ml_run_id=9999,
            open_ml_flow_name="test.classifier",
            accuracy=0.95
        )
        
        # Insert entry
        entry_id = repo.insert_entry(test_entry)
        assert entry_id is not None, "Failed to insert entry"
        print(f"✅ Entry inserted with ID: {entry_id}")
        
        # Retrieve entry
        retrieved_entry = repo.get_by_id(entry_id)
        assert retrieved_entry is not None, "Failed to retrieve entry"
        assert retrieved_entry.open_ml_run_id == 9999, "Run ID mismatch"
        assert retrieved_entry.accuracy == 0.95, "Accuracy mismatch"
        print("✅ Entry retrieved successfully")
        
        # Test count
        count = repo.count_entries()
        assert count == 1, f"Expected 1 entry, got {count}"
        print(f"✅ Entry count correct: {count}")
        
        # Test database service
        print("\n🔧 Testing database service...")
        service = DatabaseService()
        service.connection = db  # Use our test database
        service.repository = repo
        
        assert service.initialize(), "Service initialization failed"
        print("✅ Database service initialized")
        
        # Add entry via service
        service_entry_id = service.add_knowledge_entry(
            meta_vector=[0.6, 0.7, 0.8, 0.9, 1.0],
            run_id=8888,
            flow_name="service.test.classifier",
            accuracy=0.88
        )
        assert service_entry_id is not None, "Service failed to add entry"
        print(f"✅ Service added entry with ID: {service_entry_id}")
        
        # Get stats
        stats = service.get_stats()
        assert stats['total_entries'] == 2, f"Expected 2 entries, got {stats['total_entries']}"
        print(f"✅ Service stats correct: {stats}")
        
        # Get best performers
        best = service.get_best_performers(limit=5)
        assert len(best) == 2, f"Expected 2 entries, got {len(best)}"
        assert best[0].accuracy >= best[1].accuracy, "Entries not sorted by accuracy"
        print("✅ Best performers query successful")
        
        print("\n🎉 All database tests passed!")
        
    finally:
        # Clean up
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
        print("🧹 Test database cleaned up")


if __name__ == "__main__":
    test_database()
