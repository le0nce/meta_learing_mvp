"""
Knowledge Builder Service
Service for building knowledge base from OpenML datasets
"""

# pylint: disable=cell-var-from-loop

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import openml
import pandas as pd

from app.database.models import KnowledgeBaseEntry, KnowledgeBaseRepository
from app.utils.utils import (
    ADDITIONAL_METRICS,
    DEFAULT_PRIMARY_METRIC,
    parse_algo_class,
    with_retry,
)


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
            "sklearn.discriminant_analysis.",
            "sklearn.ensemble.",
            "sklearn.gaussian_process.",
            "sklearn.naive_bayes.",
            "sklearn.neighbors.",
            "sklearn.neural_network.",
            "sklearn.svm.",
            "sklearn.tree.",
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

    def update_knowledge_base(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements,unused-argument
        self,
        *,
        primary_metric: str = DEFAULT_PRIMARY_METRIC,
        extra_metrics: Optional[Sequence[str]] = tuple(ADDITIONAL_METRICS),
        store_top_n: int = 1,
        limit_dataset_n: Optional[int] = 10,
        dataset_sample_seed: int = 42,
        min_algorithms_per_dataset: int = 5,
        include_ensembles: bool = True,
        destructive_rebuild: bool = True,
    ) -> None:
        logger = self.logger
        metrics = [primary_metric] + [
            m for m in (extra_metrics or []) if m != primary_metric
        ]

        if destructive_rebuild:
            cleared = self.repository.clear_all_entries()
            logger.warning("Cleared knowledge base: %s entries removed", cleared)

        # Get all classifiaction tasks from OpenML
        clf_tasks = with_retry(
            lambda: openml.tasks.list_tasks(
                output_format="dataframe",
                task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
            ),
            logger=logger,
        ).dropna(subset=["did", "tid"])

        grouped = clf_tasks.groupby("did")["tid"].apply(list)

        # Limit to a specific number of datasets if requested
        if limit_dataset_n is not None:
            rng = np.random.default_rng(dataset_sample_seed)
            dids = grouped.index.to_numpy()
            if len(dids) > limit_dataset_n:
                picked = set(rng.choice(dids, size=limit_dataset_n, replace=False))
                grouped = grouped.loc[grouped.index.isin(picked)]

        dataset_task_map: Dict[int, int] = {}
        for dataset_id, tasks in grouped.items():  # pylint: disable=cell-var-from-loop
            try:
                def get_evaluations():  # pylint: disable=cell-var-from-loop
                    return openml.evaluations.list_evaluations(
                        function=primary_metric,
                        output_format="dataframe",
                        tasks=tasks,
                    )

                evals_df = with_retry(get_evaluations, logger=logger)
                if evals_df is None or evals_df.empty:
                    logger.warning(
                        "No evaluations found for dataset %s with tasks %s",
                        dataset_id, tasks
                    )
                    continue
                counts = (
                    evals_df.groupby("task_id").size().reset_index(name="eval_count")
                )
                if counts.empty:
                    logger.warning(
                        "No evaluations found for dataset %s with tasks %s",
                        dataset_id, tasks
                    )
                    continue
                counts = counts.sort_values(
                    ["eval_count", "task_id"], ascending=[False, True]
                )
                dataset_task_map[dataset_id] = int(counts.iloc[0]["task_id"])
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Failed to get evaluations for dataset %s with tasks %s: %s",
                    dataset_id, tasks, e
                )
                continue

        all_leaderboards: List[pd.DataFrame] = []
        for dataset_id, task_id in dataset_task_map.items():  # pylint: disable=cell-var-from-loop
            frames = []
            for m in metrics:  # pylint: disable=cell-var-from-loop
                try:
                    def get_metric_evaluations():  # pylint: disable=cell-var-from-loop
                        return openml.evaluations.list_evaluations(
                            function=m,
                            output_format="dataframe",
                            tasks=[task_id],
                        )

                    df = with_retry(get_metric_evaluations, logger=logger)
                    if df is None or df.empty:
                        logger.warning(
                            "No evaluations found for task %s with metric %s",
                            task_id, m
                        )
                        continue
                    df = df[
                        [
                            "setup_id",
                            "flow_id",
                            "flow_name",
                            "task_id",
                            "data_id",
                            "data_name",
                            "value",
                            "run_id",
                        ]
                    ].copy()
                    df["metric"] = m
                    frames.append(df)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Failed to get evaluations for task %s with metric %s: %s",
                        task_id, m, e
                    )
                    continue
            if not frames:
                continue

            evals_long = pd.concat(frames, ignore_index=True)
            evals_long = evals_long[evals_long["flow_name"].apply(self._is_base_learner)]
            if evals_long.empty:
                logger.warning(
                    "No valid base learners found for task %s in dataset %s",
                    task_id, dataset_id
                )
                continue

            grp_cols = [
                "setup_id",
                "flow_id",
                "flow_name",
                "task_id",
                "data_id",
                "data_name",
                "metric",
            ]
            per_setup = evals_long.groupby(grp_cols, as_index=False)["value"].mean()
            per_setup["algo_family"] = per_setup["flow_name"].map(parse_algo_class)

            per_family = per_setup.sort_values(
                "value", ascending=False
            ).drop_duplicates(subset=["data_id", "task_id", "metric", "algo_family"])

            wide = per_family.pivot_table(
                index=[
                    "data_id",
                    "task_id",
                    "data_name",
                    "algo_family",
                    "flow_id",
                    "setup_id",
                    "flow_name",
                ],
                columns="metric",
                values="value",
                aggfunc="first",
            ).reset_index()

            if primary_metric not in wide.columns:
                continue
            wide = wide.dropna(subset=[primary_metric])
            if wide["algo_family"].nunique() < min_algorithms_per_dataset:
                logger.warning(
                    "Not enough algorithms for dataset %s with task %s",
                    dataset_id, task_id
                )
                continue

            all_leaderboards.append(wide)

            topn = (
                wide.sort_values(primary_metric, ascending=False)
                .head(store_top_n)
                .to_dict(orient="records")
            )
            for rec in topn:
                metrics_dict = {
                    m: float(rec[m]) for m in metrics if m in rec and pd.notna(rec[m])
                }
                entry = KnowledgeBaseEntry(
                    run_id=int(rec.get("run_id", 0)),  # Use 0 if run_id is missing
                    task_id=int(rec["task_id"]),
                    setup_id=int(rec["setup_id"]),
                    flow_id=int(rec["flow_id"]),
                    flow_name=str(rec["flow_name"]),
                    algo_family=str(rec["algo_family"]),
                    data_id=int(rec["data_id"]),
                    data_name=str(rec["data_name"]),
                    metrics=metrics_dict,
                    meta_vector=None,
                )
                self.repository.insert_entry(entry)

    def get_knowledge_base_stats(self) -> List[KnowledgeBaseEntry]:
        try:
            return self.repository.get_all()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to retrieve knowledge base stats: %s", e)
            return []

    def get_base_learner_summary(self) -> dict:
        try:
            entries = self.repository.get_all()
            if not entries:
                return {
                    "total_entries": 0,
                    "unique_algorithms": 0,
                    "algorithm_counts": {},
                    "most_common_algorithm": None,
                    "datasets_covered": 0,
                    "tasks_covered": 0,
                }
            algo_families = [
                getattr(e, "algo_family", None)
                for e in entries
                if getattr(e, "algo_family", None)
            ]
            counts: Dict[str, int] = {}
            for fam in algo_families:
                counts[fam] = counts.get(fam, 0) + 1
            sorted_flows = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return {
                "total_entries": len(entries),
                "unique_algorithms": len(counts),
                "algorithm_counts": dict(sorted_flows),
                "most_common_algorithm": sorted_flows[0][0] if sorted_flows else None,
                "datasets_covered": len(set(e.data_id for e in entries)),
                "tasks_covered": len(set(e.task_id for e in entries)),
            }
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to get base learner summary: %s", e)
            return {"error": str(e)}
