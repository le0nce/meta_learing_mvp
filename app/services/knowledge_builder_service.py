"""
Knowledge Builder Service
Service for building knowledge base from OpenML datasets
"""

import logging
from typing import List
import numpy as np
import openml

from app.database.models import KnowledgeBaseRepository, KnowledgeBaseEntry
from app.services.console_protocol import ConsoleProtocol


class KnowledgeBuilderService:
    """Service for building knowledge base from OpenML datasets"""

    def __init__(self):
        """
        Initialize the knowledge builder service
        """
        self.logger = logging.getLogger(__name__)
        self.repository = KnowledgeBaseRepository()

    def update_knowledge_base(self) -> None:
        tasks_df = openml.tasks.list_tasks(output_format="dataframe")
        clf_tasks = tasks_df[tasks_df['task_type'] == 'Supervised Classification']

        # Step 2: Group by dataset to get ALL associated task IDs
        task_map = clf_tasks.groupby('did')['tid'].apply(list)

        # Step 3: For each dataset, find best-performing algorithm across all its tasks
        results = []
        for did, task_ids in task_map.items():
            try:
                dataset = openml.datasets.get_dataset(did, download_all_files=False)
                dataset_name = dataset.name

                best_prec = -1
                best_algorithm = None
                best_tid = None

                for tid in task_ids:
                    try:
                        runs_df = openml.runs.list_runs(task=[tid], size=1000, output_format="dataframe")
                        if runs_df.empty:
                            continue

                        for run_id in runs_df['run_id']:
                            try:
                                run = openml.runs.get_run(run_id)
                                prec = run.evaluations.get('precision', None)
                                if prec is not None and prec > best_prec:
                                    best_prec = prec
                                    best_algorithm = run.flow_name
                                    best_tid = tid
                            except:
                                continue
                    except:
                        continue

                if best_algorithm:
                    results.append({
                        "Dataset Name": dataset_name,
                        "Dataset ID": did,
                        "Best Task ID": best_tid,
                        "Best Algorithm": best_algorithm,
                        "Accuracy": best_accuracy
                    })

            except Exception:
                continue


    def get_knowledge_base_stats(self) -> List[KnowledgeBaseEntry]:
        """
        Get statistics of the knowledge base
        
        Returns:
            List of KnowledgeBaseEntry objects with stats
        """
        try:
            entries: List[KnowledgeBaseEntry] = self.repository.get_all()
            if not entries:
                self.logger.info("Knowledge base is empty.")
                return []

            self.logger.info(f"Retrieved {len(entries)} entries from knowledge base.")
            return entries
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge base stats: {e}")
            return []