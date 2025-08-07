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
from collections import defaultdict

class KnowledgeBuilderService:
    """Service for building knowledge base from OpenML datasets"""

    def __init__(self):
        """
        Initialize the knowledge builder service
        """
        self.logger = logging.getLogger(__name__)
        self.repository = KnowledgeBaseRepository()

    def update_knowledge_base(self) -> None:
        # 1. Load all classification tasks from OpenML
        clf_tasks = openml.tasks.list_tasks(output_format="dataframe", task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION)

        # 2. Drop tasks without dataset id (edge case)
        clf_tasks = clf_tasks.dropna(subset=["did", "tid"])

        # 3. Aggregate tasks by dataset id
        tasks_by_dataset = clf_tasks.groupby("did")["tid"].apply(list)
        self.logger.info(f"Found {len(tasks_by_dataset)} datasets with classification tasks.")
        dataset_task_map = {}
        # Limit to first 10 datasets for demonstration
        for dataset_id, tasks in list(tasks_by_dataset.items())[:10]:
            self.logger.info(f"Dataset {dataset_id} has {len(tasks)} classification tasks.")
            evals_df = openml.evaluations.list_evaluations(
                function='predictive_accuracy',
                output_format='dataframe',
                tasks=tasks
            )

            # Count evaluations per task for this dataset
            task_eval_counts = evals_df.groupby('task_id').size().reset_index(name='eval_count')
            
            # Find the task with the most evaluations
            most_popular_task = task_eval_counts.loc[task_eval_counts['eval_count'].idxmax()]
            
            dataset_task_map[dataset_id] = most_popular_task['task_id']
            self.logger.info(f"Most popular task for dataset {dataset_id}: {most_popular_task['task_id']} ({most_popular_task['eval_count']} evaluations)")

        # For each most popular dataset supervised classifcation task, fetch top N runs
        for dataset_id, task_id in dataset_task_map.items():
            self.logger.info(f"Fetching top runs for dataset {dataset_id}, task {task_id}")
            
            # Get all available evaluations for this task
            task_evaluations = openml.evaluations.list_evaluations(
                function='predictive_accuracy',
                output_format='dataframe',
                tasks=[task_id]
            )
            
            self.logger.info(f"Retrieved {len(task_evaluations)} evaluations for task {task_id}")
            self.logger.info(f"Max predictive accuracy: {task_evaluations['value'].max():.6f}")
            self.logger.info(f"Min predictive accuracy: {task_evaluations['value'].min():.6f}")
            self.logger.info(f"Mean predictive accuracy: {task_evaluations['value'].mean():.6f}")
            
            # Count high-performing evaluations
            high_perf = len(task_evaluations[task_evaluations['value'] > 0.99])
            very_high_perf = len(task_evaluations[task_evaluations['value'] > 0.999])
            self.logger.info(f"Evaluations > 0.99: {high_perf}, > 0.999: {very_high_perf}")
            
            # Get top evaluations
            top_5_evaluations = task_evaluations.nlargest(10, 'value')
            self.logger.info(f"Top 10 accuracy values: {top_5_evaluations['value'].tolist()}")
            
            # Process each run
            for _, run in top_5_evaluations.iterrows():
                run_to_store = KnowledgeBaseEntry(
                    run_id=run['run_id'],
                    task_id=task_id,
                    setup_id=run['setup_id'],
                    flow_id=run['flow_id'],
                    flow_name=run['flow_name'],
                    data_id=dataset_id,
                    data_name=run['data_name'],
                    eval_metric='predictive_accuracy',
                    eval_value=run['value'],
                    meta_vector=None  # Placeholder for now, can be calculated later
                )
                self.logger.info(f"Storing run {run['run_id']} for dataset {dataset_id}")
                entry_id = self.repository.insert_entry(run_to_store)
                if entry_id:
                    self.logger.info(f"Inserted entry with ID {entry_id} for run {run['run_id']}")
                else:
                    self.logger.error(f"Failed to insert entry for run {run['run_id']}")
        self.logger.info("Knowledge base update completed.")
        

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