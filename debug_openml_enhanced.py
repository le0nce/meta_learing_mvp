#!/usr/bin/env python3
"""
Enhanced debug script to investigate OpenML API pagination and parameters
"""

import openml
import pandas as pd

def debug_with_different_parameters():
    """Try different API parameters to get all evaluations"""
    print("=== Testing Different API Parameters ===")
    
    # Test 1: Default call
    print("1. Default call:")
    evals_default = openml.evaluations.list_evaluations(
        function='predictive_accuracy',
        output_format='dataframe',
        tasks=[2]
    )
    print(f"   Results: {len(evals_default)} evaluations, max value: {evals_default['value'].max()}")
    
    # Test 2: With larger size
    print("2. With size=10000:")
    try:
        evals_large = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            output_format='dataframe',
            tasks=[2],
            size=10000
        )
        print(f"   Results: {len(evals_large)} evaluations, max value: {evals_large['value'].max()}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 3: With offset
    print("3. With offset=0, size=5000:")
    try:
        evals_offset = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            output_format='dataframe',
            tasks=[2],
            offset=0,
            size=5000
        )
        print(f"   Results: {len(evals_offset)} evaluations, max value: {evals_offset['value'].max()}")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test 4: Check if there are more recent results
    print("4. Checking most recent evaluations (sorted by upload_time):")
    try:
        # Sort by upload_time to see most recent
        recent_evals = evals_default.sort_values('upload_time', ascending=False)
        print(f"   Most recent upload time: {recent_evals.iloc[0]['upload_time']}")
        print(f"   Top 5 most recent evaluations:")
        print(recent_evals[['run_id', 'value', 'upload_time', 'flow_name']].head())
        
        # Check if recent evaluations have higher values
        recent_top_10 = recent_evals.head(100)  # Get top 100 most recent
        recent_max = recent_top_10['value'].max()
        print(f"   Max value in 100 most recent: {recent_max}")
        
    except Exception as e:
        print(f"   Failed: {e}")

def check_specific_high_performance_runs():
    """Look for specific runs that might have higher performance"""
    print("\n=== Searching for High Performance Runs ===")
    
    # Try to get runs directly
    try:
        print("1. Trying to get runs for task 2 directly:")
        runs = openml.runs.list_runs(task=[2], output_format='dataframe')
        print(f"   Found {len(runs)} runs")
        
        if len(runs) > 0:
            print("   Columns in runs:", runs.columns.tolist())
            # If there are evaluation columns, check them
            eval_cols = [col for col in runs.columns if 'accuracy' in col.lower() or 'predictive' in col.lower()]
            if eval_cols:
                print(f"   Evaluation columns found: {eval_cols}")
                for col in eval_cols:
                    if runs[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        print(f"   Max value in {col}: {runs[col].max()}")
                        
    except Exception as e:
        print(f"   Failed to get runs: {e}")
    
    # Try to search for evaluations with different criteria
    print("\n2. Trying to get ALL evaluations (no size limit):")
    try:
        all_evals = openml.evaluations.list_evaluations(
            function='predictive_accuracy',
            tasks=[2],
            output_format='dataframe'
        )
        print(f"   Total evaluations: {len(all_evals)}")
        print(f"   Max value: {all_evals['value'].max()}")
        print(f"   Values > 0.997: {len(all_evals[all_evals['value'] > 0.997])}")
        print(f"   Values > 0.999: {len(all_evals[all_evals['value'] > 0.999])}")
        
        # Check if there are any very high values
        very_high = all_evals[all_evals['value'] > 0.998]
        if len(very_high) > 0:
            print("   Runs with value > 0.998:")
            print(very_high[['run_id', 'value', 'flow_name', 'upload_time']])
            
    except Exception as e:
        print(f"   Failed: {e}")

def investigate_api_limitations():
    """Investigate potential API limitations or caching issues"""
    print("\n=== Investigating API Limitations ===")
    
    # Check if there are different evaluation functions
    print("1. Checking available evaluation functions:")
    try:
        # This might not work, but let's try
        functions = openml.evaluations.list_evaluation_measures()
        print(f"   Available functions: {functions}")
    except Exception as e:
        print(f"   Could not list functions: {e}")
    
    # Try getting evaluations for a different function to compare
    print("\n2. Trying different evaluation functions:")
    alt_functions = ['area_under_roc_curve', 'f_measure', 'precision', 'recall']
    
    for func in alt_functions:
        try:
            alt_evals = openml.evaluations.list_evaluations(
                function=func,
                tasks=[2],
                output_format='dataframe',
                size=100  # Small sample
            )
            print(f"   {func}: {len(alt_evals)} evaluations, max: {alt_evals['value'].max()}")
        except Exception as e:
            print(f"   {func}: Failed - {e}")

if __name__ == "__main__":
    debug_with_different_parameters()
    check_specific_high_performance_runs() 
    investigate_api_limitations()
