#!/usr/bin/env python3
"""
Specific test for task 2 to verify the highest accuracy values
"""

import openml
import pandas as pd

def test_task_2_comprehensive():
    """Comprehensive test for task 2"""
    print("=== Comprehensive Task 2 Analysis ===")
    
    # Get all evaluations
    evals_df = openml.evaluations.list_evaluations(
        function='predictive_accuracy',
        output_format='dataframe',
        tasks=[2]
    )
    
    print(f"Total evaluations: {len(evals_df)}")
    print(f"Max value: {evals_df['value'].max()}")
    
    # Sort by value descending and show top 20
    top_evals = evals_df.sort_values('value', ascending=False)
    
    print("\n=== Top 20 Evaluations by Predictive Accuracy ===")
    print(top_evals[['run_id', 'value', 'flow_name', 'upload_time']].head(20))
    
    # Look specifically for values > 0.999
    high_acc = evals_df[evals_df['value'] > 0.999]
    print(f"\n=== Evaluations with accuracy > 0.999 ===")
    print(f"Count: {len(high_acc)}")
    if len(high_acc) > 0:
        print(high_acc[['run_id', 'value', 'flow_name', 'upload_time']])
    
    # Check recent high-performing runs
    recent_evals = evals_df[evals_df['upload_time'] >= '2020-01-01'].sort_values('value', ascending=False)
    print(f"\n=== Recent evaluations (2020+) - Top 10 ===")
    print(f"Total recent evaluations: {len(recent_evals)}")
    if len(recent_evals) > 0:
        print(recent_evals[['run_id', 'value', 'flow_name', 'upload_time']].head(10))
    
    # Test manual retrieval of specific high-scoring run
    print("\n=== Testing manual run retrieval ===")
    top_run_id = top_evals.iloc[0]['run_id']
    try:
        run_details = openml.runs.get_run(top_run_id)
        print(f"Top run {top_run_id} details:")
        print(f"  Flow ID: {run_details.flow_id}")
        print(f"  Task ID: {run_details.task_id}")
        if hasattr(run_details, 'evaluations') and run_details.evaluations:
            for metric, value in run_details.evaluations.items():
                if 'accuracy' in metric.lower():
                    print(f"  {metric}: {value}")
    except Exception as e:
        print(f"Failed to get run details: {e}")

if __name__ == "__main__":
    test_task_2_comprehensive()
