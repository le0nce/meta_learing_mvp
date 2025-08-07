#!/usr/bin/env python3
"""
Debug script to investigate OpenML API vs Website discrepancies
"""

import openml
import pandas as pd

def debug_task_evaluations():
    """Debug task 2 evaluations"""
    print("=== Debugging Task 2 Evaluations ===")
    
    # Get evaluations for task 2
    print("Fetching evaluations for task 2...")
    evals_df = openml.evaluations.list_evaluations(
        function='predictive_accuracy',
        output_format='dataframe',
        tasks=[2]
    )
    
    print(f"Total evaluations found: {len(evals_df)}")
    print(f"Columns: {evals_df.columns.tolist()}")
    
    # Show some basic stats
    print(f"\nValue statistics:")
    print(f"Min: {evals_df['value'].min()}")
    print(f"Max: {evals_df['value'].max()}")
    print(f"Mean: {evals_df['value'].mean()}")
    
    # Show top 10 by value
    print(f"\n=== Top 10 evaluations by value ===")
    top_10 = evals_df.nlargest(10, 'value')
    print(top_10[['run_id', 'task_id', 'setup_id', 'flow_id', 'flow_name', 'value']])
    
    # Check if there are any NaN values
    print(f"\nNaN values in 'value' column: {evals_df['value'].isna().sum()}")
    
    # Check unique flow names in top results
    print(f"\n=== Flow names in top 20 results ===")
    top_20 = evals_df.nlargest(20, 'value')
    print(top_20['flow_name'].value_counts())
    
    # Try different approaches to get evaluations
    print(f"\n=== Alternative API calls ===")
    
    # Try with different parameters
    try:
        print("Trying to get evaluations with different size parameter...")
        evals_alt = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            output_format='dataframe',
            tasks=[2],
            size=1000
        )
        print(f"Alternative call returned {len(evals_alt)} evaluations")
        print(f"Max value in alternative call: {evals_alt['value'].max()}")
        
        if len(evals_alt) != len(evals_df):
            print("WARNING: Different number of evaluations returned!")
        
    except Exception as e:
        print(f"Alternative call failed: {e}")
    
    # Try to get specific run information
    print(f"\n=== Checking specific runs ===")
    print("Looking for runs with predictive_accuracy > 0.999...")
    high_perf_runs = evals_df[evals_df['value'] > 0.999]
    print(f"Found {len(high_perf_runs)} runs with value > 0.999")
    if len(high_perf_runs) > 0:
        print(high_perf_runs[['run_id', 'task_id', 'setup_id', 'flow_id', 'flow_name', 'value']])
    
    # Check if there are duplicates
    print(f"\n=== Checking for duplicates ===")
    print(f"Unique run_ids: {evals_df['run_id'].nunique()}")
    print(f"Total rows: {len(evals_df)}")
    
    if evals_df['run_id'].nunique() != len(evals_df):
        print("WARNING: Duplicate run_ids found!")
        duplicates = evals_df[evals_df.duplicated(subset=['run_id'], keep=False)]
        print("Duplicate run_ids:")
        print(duplicates[['run_id', 'value']].sort_values('run_id'))

def debug_task_info():
    """Get information about task 2"""
    print("\n=== Task 2 Information ===")
    
    try:
        task = openml.tasks.get_task(2)
        print(f"Task ID: {task.task_id}")
        print(f"Task type: {task.task_type}")
        print(f"Dataset ID: {task.dataset_id}")
        
        # Get dataset info
        dataset = openml.datasets.get_dataset(task.dataset_id)
        print(f"Dataset name: {dataset.name}")
        print(f"Dataset description: {dataset.description[:200]}...")
        
    except Exception as e:
        print(f"Failed to get task info: {e}")

if __name__ == "__main__":
    debug_task_evaluations()
    debug_task_info()
