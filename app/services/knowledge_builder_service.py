"""
Knowledge Builder Service
Service for building knowledge base from OpenML datasets
"""

import logging
from typing import List
import numpy as np
import openml

from app.database.models import KnowledgeBaseRepository, KnowledgeBaseEntry

class KnowledgeBuilderService:
    """Service for building knowledge base from OpenML datasets"""

    def __init__(self):
        """
        Initialize the knowledge builder service
        """
        self.logger = logging.getLogger(__name__)
        self.repository = KnowledgeBaseRepository()
        
        # Define the fixed set of allowed base learners (sklearn only)
        # Based on scikit-learn's standard classifier comparison
        # If the flow name starts with one of these, we consider it a base learner
        self.allowed_modules = [
            'sklearn.discriminant_analysis.',
            'sklearn.ensemble.',
            'sklearn.gaussian_process.',
            'sklearn.naive_bayes.',
            'sklearn.neighbors.',
            'sklearn.neural_network.',
            'sklearn.svm.',
            'sklearn.tree.'
        ]

    def _is_base_learner(self, flow_name: str) -> bool:
        """
        Check if a flow represents one of our allowed base learner algorithms
        
        Args:
            flow_name: Name of the flow/algorithm
            
        Returns:
            True if it's in our allowed list, False otherwise
        """
        # First, reject anything that looks like a pipeline or complex workflow
        flow_name_lower = flow_name.lower()
        if not flow_name_lower.startswith(tuple(self.allowed_modules)):
            return False

        return True

    def update_knowledge_base(self, evaluation_metrics: str = 'predictive_accuracy', store_top_n: int = 1, limit_dataset_n: int = None) -> None:
        """
        Update knowledge base with fresh data from OpenML
        WARNING: This will clear ALL existing knowledge base entries first!
        
        Args:
            evaluation_metrics: Evaluation metric to use for ranking
            store_top_n: Number of top evaluations to store per dataset
            limit_dataset_n: Maximum number of datasets to process
        """
        self.logger.info("Starting knowledge base update - WARNING: This will clear all existing entries!")
        
        # Clear existing knowledge base entries first
        try:
            cleared_count = self.repository.clear_all_entries()
            self.logger.info(f"Cleared {cleared_count} existing entries from knowledge base")
        except Exception as e:
            self.logger.error(f"Failed to clear existing entries: {e}")
            raise
        
        # 1. Load all classification tasks from OpenML
        clf_tasks = openml.tasks.list_tasks(output_format="dataframe", task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION)

        # 2. Drop tasks without dataset id (edge case)
        clf_tasks = clf_tasks.dropna(subset=["did", "tid"])

        # 3. Aggregate tasks by dataset id
        if limit_dataset_n is not None:
            tasks_by_dataset = clf_tasks.groupby("did")["tid"].apply(list)[:limit_dataset_n]
            self.logger.info(f"Limiting to {limit_dataset_n} datasets with classification tasks")
        else:
            tasks_by_dataset = clf_tasks.groupby("did")["tid"].apply(list)
            self.logger.info("Processing all datasets with classification tasks")
        
        self.logger.info(f"Found {len(tasks_by_dataset)} datasets with classification tasks.")

        # 4. For each dataset, find the most popular supervised classification task
        dataset_task_map = {}
        for dataset_id, tasks in list(tasks_by_dataset.items()):
            self.logger.info(f"Dataset {dataset_id} has {len(tasks)} classification tasks.")
            try:
                evals_df = openml.evaluations.list_evaluations(
                    function=evaluation_metrics,
                    output_format='dataframe',
                    tasks=tasks
                )

                # Check if we got any evaluations back
                if evals_df.empty:
                    self.logger.warning(f"No evaluations found for dataset {dataset_id} tasks {tasks}, skipping")
                    continue

                # Count evaluations per task for this dataset
                task_eval_counts = evals_df.groupby('task_id').size().reset_index(name='eval_count')
                
                # Check if we have any task evaluation counts
                if task_eval_counts.empty:
                    self.logger.warning(f"No task evaluation counts for dataset {dataset_id}, skipping")
                    continue
                
                # Find the task with the most evaluations
                most_popular_task = task_eval_counts.loc[task_eval_counts['eval_count'].idxmax()]
                
                # Map dataset to its most popular task
                dataset_task_map[dataset_id] = int(most_popular_task['task_id'])
                self.logger.info(f"Most popular task for dataset {dataset_id}: {most_popular_task['task_id']} ({most_popular_task['eval_count']} evaluations)")
            
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_id}: {e}")
                continue

        # 5. For each most popular dataset supervised classification task, fetch top N evaluations from base learners only
        for dataset_id, task_id in dataset_task_map.items():
            self.logger.info(f"Fetching top evaluations for dataset {dataset_id}, task {task_id}")

            try:
                # Get all available evaluations for this task
                task_evaluations = openml.evaluations.list_evaluations(
                    function=evaluation_metrics,
                    output_format='dataframe',
                    tasks=[task_id]
                )
                
                # Check if we got any evaluations back
                if task_evaluations.empty:
                    self.logger.warning(f"No evaluations found for task {task_id}, skipping")
                    continue
                
                self.logger.info(f"Retrieved {len(task_evaluations)} evaluations for task {task_id}")
                
                # Filter for base learners only
                base_learner_evaluations = task_evaluations[
                    task_evaluations['flow_name'].apply(self._is_base_learner)
                ]
                
                self.logger.info(f"After filtering for base learners: {len(base_learner_evaluations)} evaluations")
                
                if len(base_learner_evaluations) == 0:
                    self.logger.warning(f"No base learner evaluations found for task {task_id}, skipping")
                    continue

                # Get top evaluations from base learners only
                top_n_evaluation = base_learner_evaluations.nlargest(store_top_n, 'value')
                self.logger.info(f"Top {store_top_n} base learner accuracy values: {top_n_evaluation['value'].tolist()}")

                # Log the flow names of top performers
                unique_flows = top_n_evaluation['flow_name'].unique()
                self.logger.info(f"Top performing base learner flows: {list(unique_flows)}")
                
                # Process each run
                for _, run in top_n_evaluation.iterrows():
                    
                    run_to_store = KnowledgeBaseEntry(
                        run_id=run['run_id'],
                        task_id=task_id,
                        setup_id=run['setup_id'],
                        flow_id=run['flow_id'],
                        flow_name=run['flow_name'],
                        data_id=dataset_id,
                        data_name=run['data_name'],
                        eval_metric=evaluation_metrics,
                        eval_value=run['value'],
                        meta_vector=None  # Placeholder for now, can be calculated later
                    )
                    self.logger.info(f"Storing {run['flow_name']} run {run['run_id']} for dataset {dataset_id}")
                    entry_id = self.repository.insert_entry(run_to_store)
                    if entry_id:
                        self.logger.info(f"Inserted entry with ID {entry_id} for run {run['run_id']}")
                    else:
                        self.logger.error(f"Failed to insert entry for run {run['run_id']}")
                        
            except Exception as e:
                self.logger.error(f"Error processing task {task_id} for dataset {dataset_id}: {e}")
                continue
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
    
    def get_base_learner_summary(self) -> dict:
        """
        Get a summary of base learners in the knowledge base
        
        Returns:
            Dictionary with summary statistics about base learners
        """
        try:
            entries = self.repository.get_all()
            if not entries:
                return {"total_entries": 0, "unique_algorithms": 0, "algorithms": []}
            
            # Count unique flow names (algorithms)
            flow_names = [entry.flow_name for entry in entries]
            unique_flows = list(set(flow_names))
            
            # Count entries per algorithm
            flow_counts = {}
            for flow_name in flow_names:
                flow_counts[flow_name] = flow_counts.get(flow_name, 0) + 1
            
            # Sort by count
            sorted_flows = sorted(flow_counts.items(), key=lambda x: x[1], reverse=True)
            
            summary = {
                "total_entries": len(entries),
                "unique_algorithms": len(unique_flows),
                "algorithm_counts": dict(sorted_flows),
                "most_common_algorithm": sorted_flows[0][0] if sorted_flows else None,
                "datasets_covered": len(set(entry.data_id for entry in entries)),
                "tasks_covered": len(set(entry.task_id for entry in entries))
            }
            
            self.logger.info(f"Knowledge base contains {summary['total_entries']} entries from {summary['unique_algorithms']} unique base learners")
            self.logger.info(f"Most common algorithm: {summary['most_common_algorithm']}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate base learner summary: {e}")
            return {"error": str(e)}