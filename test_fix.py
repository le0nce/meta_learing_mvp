#!/usr/bin/env python3
"""
Simple test script to verify the knowledge base stats fix
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.knowledge_builder_service import KnowledgeBuilderService

def test_knowledge_base_stats():
    """Test that get_knowledge_base_stats works without errors"""
    try:
        service = KnowledgeBuilderService()
        stats = service.get_knowledge_base_stats()
        print(f"✅ Successfully retrieved knowledge base stats: {len(stats)} entries found")
        
        # Also test the summary method
        summary = service.get_base_learner_summary()
        print(f"✅ Successfully retrieved base learner summary: {summary}")
        
        return True
    except Exception as e:
        print(f"❌ Error retrieving knowledge base stats: {e}")
        return False

if __name__ == "__main__":
    print("Testing knowledge base stats fix...")
    success = test_knowledge_base_stats()
    sys.exit(0 if success else 1)
