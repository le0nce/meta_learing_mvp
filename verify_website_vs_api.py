#!/usr/bin/env python3
"""
Verification script to check if the OpenML website shows different results
than the API for task 2 dataset 2
"""

import openml
import requests
import pandas as pd
from urllib.parse import urljoin

def check_website_vs_api():
    """Compare website results with API results"""
    print("=== Comparing OpenML Website vs API ===")
    
    # API Results
    print("1. API Results:")
    api_evals = openml.evaluations.list_evaluations(
        function='predictive_accuracy',
        output_format='dataframe',
        tasks=[2]
    )
    
    api_max = api_evals['value'].max()
    print(f"   API Max Accuracy: {api_max:.6f}")
    print(f"   API Total Evaluations: {len(api_evals)}")
    
    # Get top 5 from API
    api_top_5 = api_evals.nlargest(5, 'value')
    print("   API Top 5 Accuracies:")
    for idx, row in api_top_5.iterrows():
        print(f"     Run {row['run_id']}: {row['value']:.6f} ({row['flow_name']})")
    
    # Attempt to check website data (this might not work directly)
    print("\n2. Website Verification:")
    print("   Note: Direct website scraping is not recommended and may not work")
    print("   You mentioned seeing 0.999688 on the OpenML website")
    print("   This could be due to:")
    print("   - Website caching issues")
    print("   - Different evaluation metrics being displayed") 
    print("   - Website showing aggregated/processed data")
    print("   - API filtering certain results")
    
    # Check if we can find any runs with the specific accuracy you mentioned
    print(f"\n3. Searching for runs with accuracy close to 0.999688:")
    target_accuracy = 0.999688
    close_matches = api_evals[abs(api_evals['value'] - target_accuracy) < 0.001]
    
    if len(close_matches) > 0:
        print(f"   Found {len(close_matches)} runs close to target accuracy:")
        for idx, row in close_matches.iterrows():
            print(f"     Run {row['run_id']}: {row['value']:.6f}")
    else:
        print(f"   No runs found with accuracy close to {target_accuracy}")
    
    # Check different ways the website might calculate or display results
    print(f"\n4. Alternative Explanations:")
    print("   a) Different rounding: API max rounded: {:.3f}".format(api_max))
    print("   b) Different task: Website might show results for a different task")
    print("   c) Different metric: Website might show AUC or F1 instead of accuracy")
    
    # Check AUC values for comparison
    try:
        auc_evals = openml.evaluations.list_evaluations(
            function='area_under_roc_curve',
            output_format='dataframe',
            tasks=[2],
            size=100  # Limit size for faster response
        )
        auc_max = auc_evals['value'].max()
        print(f"   d) Max AUC (might be what website shows): {auc_max:.6f}")
        
        # Check if AUC max is close to what you saw
        if abs(auc_max - 0.999688) < 0.01:
            print("   *** POSSIBLE MATCH: Website might be showing AUC instead of accuracy! ***")
            
    except Exception as e:
        print(f"   Could not fetch AUC data: {e}")

def recommend_solution():
    """Recommend what to do about this discrepancy"""
    print(f"\n=== Recommendations ===")
    print("1. Your code is working correctly - it gets the highest accuracy from the API")
    print("2. The discrepancy with the website (0.999688 vs 0.996659) could be due to:")
    print("   - Website showing a different metric (like AUC)")
    print("   - Website caching/display issues")
    print("   - API filtering or different data access")
    print("3. Solutions:")
    print("   - Trust the API data as it's more reliable for programmatic access")
    print("   - Contact OpenML support if the discrepancy is critical")
    print("   - Check if the website is showing AUC instead of accuracy")
    print("   - Verify which task ID the website is actually displaying")

if __name__ == "__main__":
    check_website_vs_api()
    recommend_solution()
