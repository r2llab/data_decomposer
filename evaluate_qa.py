#!/usr/bin/env python3
"""
Evaluation script for the ReSP QA system.
This script runs questions from a ground truth file through the ReSP system,
compares the answers to the ground truth, and generates evaluation metrics.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from core.factory import ImplementationFactory
from core.config import ConfigurationManager
# Import implementations to register them
import implementations
import openai
import re

# Optional: Import BERT-based metrics if available
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("BERTScore not available. Install with: pip install bert-score")

def load_ground_truth(gt_file: str) -> List[Dict[str, str]]:
    """Load ground truth data from CSV file."""
    gt_data = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data.append(row)
    return gt_data

def calculate_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores between hypothesis and reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bert_score(hypothesis: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore between hypothesis and reference."""
    if not BERT_SCORE_AVAILABLE:
        return {'P': 0.0, 'R': 0.0, 'F1': 0.0}
    
    P, R, F1 = bert_score([hypothesis], [reference], lang='en', rescale_with_baseline=True)
    return {
        'P': P.item(),
        'R': R.item(),
        'F1': F1.item()
    }

def calculate_llm_correctness(hypothesis: str, reference: str, question: str) -> float:
    """
    Use an LLM to evaluate the correctness of the hypothesis compared to the reference.
    
    Args:
        hypothesis: The system-generated answer
        reference: The ground truth answer
        question: The original question
        api_key: OpenAI API key (optional if set as environment variable)
        
    Returns:
        A score between 0 and 1 representing correctness (1 = fully correct, 0 = incorrect)
    """
    

    
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    if not openai.api_key:
        print("No OpenAI API key provided, skipping LLM correctness evaluation")
        return 0.0
    
    try:
        # Create prompt for the LLM
        prompt = f"""
You are an expert evaluator assessing the correctness of an answer to a question.

Question: {question}

Ground Truth Answer: {reference}

System Answer: {hypothesis}

Evaluate how correct the System Answer is compared to the Ground Truth Answer.
Give a score from 0 to 1 where:
- 1.0 means the System Answer is fully correct and contains all the information from the Ground Truth
- 0.0 means the System Answer is completely incorrect
- Values between 0 and 1 indicate partial correctness

Output a single line with just the score as a decimal between 0 and 1.
"""

        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4o",  # Can be changed to a different model if needed
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Use low temperature for more consistent evaluations
            max_tokens=300
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content.strip()
        
        # Extract the score from the response - find the last number between 0 and 1
        score_matches = re.findall(r'(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)(?:$|\s)', response_text)
        if score_matches:
            score = float(score_matches[-1])  # Take the last match as the final score
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        else:
            print(f"Could not extract a score from LLM response: {response_text}")
            return 0.0
            
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return 0.0

def calculate_string_similarity(hypothesis: str, reference: str) -> float:
    """Calculate simple string similarity using SequenceMatcher."""
    return SequenceMatcher(None, hypothesis, reference).ratio()

def calculate_source_metrics(system_sources: List[str], gt_sources: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 for sources."""
    # Normalize the sources to handle path differences
    system_set = {os.path.basename(src) for src in system_sources if src}
    gt_set = {os.path.basename(src) for src in gt_sources if src}
    
    # Log for debugging
    print(f"System sources (normalized): {system_set}")
    print(f"Ground truth sources (normalized): {gt_set}")
    
    # Calculate metrics
    true_positives = len(system_set.intersection(gt_set))
    precision = true_positives / len(system_set) if system_set else 0.0
    recall = true_positives / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'overlap': true_positives,
        'system_sources_count': len(system_set),
        'gt_sources_count': len(gt_set)
    }

def run_evaluation(implementation, gt_data: List[Dict[str, str]], output_file: str):
    """Run evaluation on all questions in ground truth data."""
    results = []
    metrics = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'string_similarity': [],
        'source_precision': [],
        'source_recall': [],
        'source_f1': [],
        'llm_correctness': [],  # New metric for LLM-based correctness
    }
    
    if BERT_SCORE_AVAILABLE:
        metrics.update({
            'bert_score_P': [],
            'bert_score_R': [],
            'bert_score_F1': [],
        })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, gt_item in enumerate(gt_data):
        question = gt_item['question']
        gt_answer = gt_item['answer']
        gt_text = gt_item['text'].split(',') if gt_item['text'] else []
        gt_table = gt_item['table'].split(',') if gt_item['table'] else []
        gt_sources = gt_text + gt_table
        
        print(f"\nProcessing question {i+1}/{len(gt_data)}: {question}")
        
        # Run the question through the implementation
        start_time = time.time()
        try:
            response = implementation.process_query(question)
            processing_time = time.time() - start_time
            
            # Extract answer and sources
            system_answer = response.get('answer', '')
            system_sources = response.get('document_sources', [])
            
            # Clean up sources (remove any None or empty values)
            system_sources = [src for src in system_sources if src]
            
            # Calculate metrics
            rouge_scores = calculate_rouge(system_answer, gt_answer)
            string_sim = calculate_string_similarity(system_answer, gt_answer)
            source_metrics = calculate_source_metrics(system_sources, gt_sources)
            
            # Calculate LLM-based correctness score
            llm_correctness = 0.0
            llm_correctness = calculate_llm_correctness(
                system_answer, gt_answer, question
            )
            metrics['llm_correctness'].append(llm_correctness)
            
            item_metrics = {
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL'],
                'string_similarity': string_sim,
                'source_precision': source_metrics['precision'],
                'source_recall': source_metrics['recall'],
                'source_f1': source_metrics['f1'],
                'processing_time': processing_time,
                'llm_correctness': llm_correctness,  # Add LLM correctness score
            }
            
            if BERT_SCORE_AVAILABLE:
                bert_scores = calculate_bert_score(system_answer, gt_answer)
                item_metrics.update({
                    'bert_score_P': bert_scores['P'],
                    'bert_score_R': bert_scores['R'],
                    'bert_score_F1': bert_scores['F1']
                })
                
                # Update running metrics list
                metrics['bert_score_P'].append(bert_scores['P'])
                metrics['bert_score_R'].append(bert_scores['R'])
                metrics['bert_score_F1'].append(bert_scores['F1'])
            
            # Update running metrics lists
            metrics['rouge1'].append(rouge_scores['rouge1'])
            metrics['rouge2'].append(rouge_scores['rouge2'])
            metrics['rougeL'].append(rouge_scores['rougeL'])
            metrics['string_similarity'].append(string_sim)
            metrics['source_precision'].append(source_metrics['precision'])
            metrics['source_recall'].append(source_metrics['recall'])
            metrics['source_f1'].append(source_metrics['f1'])
            
            # Create result item
            result_item = {
                'question': question,
                'gt_answer': gt_answer,
                'system_answer': system_answer,
                'gt_sources': gt_sources,
                'system_sources': system_sources,
                'metrics': item_metrics
            }
            
            results.append(result_item)
            
            print(f"Processed in {processing_time:.2f}s")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}, String similarity: {string_sim:.4f}")
            print(f"Source F1: {source_metrics['f1']:.4f} (P: {source_metrics['precision']:.4f}, R: {source_metrics['recall']:.4f})")
            print(f"LLM Correctness: {llm_correctness:.4f}")
            
        except Exception as e:
            print(f"Error processing question: {e}")
            # Add failed item
            results.append({
                'question': question,
                'gt_answer': gt_answer,
                'system_answer': f"ERROR: {str(e)}",
                'gt_sources': gt_sources,
                'system_sources': [],
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    for metric_name, values in metrics.items():
        if values:  # Only calculate if we have values
            aggregate_metrics[metric_name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values)
            }
    
    # Add aggregate metrics to results
    final_results = {
        'individual_results': results,
        'aggregate_metrics': aggregate_metrics
    }
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {output_file}")
    print("\nAggregate metrics:")
    for metric_name, stats in aggregate_metrics.items():
        print(f"{metric_name}: mean={stats['mean']:.4f}, median={stats['median']:.4f}")

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate ReSP QA system against ground truth')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gt-file', type=str, required=True, help='Path to ground truth file')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to output file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to evaluate')
    args = parser.parse_args()
    
    # Load configuration using the same approach as main.py
    config_manager = ConfigurationManager(args.config)
    impl_config = config_manager.get_implementation_config()
    
    # Create implementation
    implementation = ImplementationFactory.create(
        impl_config['name'],
        impl_config['config']
    )
    
    # Load ground truth data
    gt_data = load_ground_truth(args.gt_file)
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        gt_data = gt_data[:args.limit]
    
    # Run evaluation
    run_evaluation(implementation, gt_data, args.output)
    
if __name__ == '__main__':
    main() 