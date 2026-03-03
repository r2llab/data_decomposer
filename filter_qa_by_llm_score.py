#!/usr/bin/env python3
"""
Script to filter question-answer pairs from a ground truth file based on LLM correctness.
Questions that can be answered correctly by an LLM without context (score > 0.5) are removed.
"""

import os
import csv
import argparse
from tqdm import tqdm
from pathlib import Path
import re
import time
import json
from openai import OpenAI

def load_ground_truth(gt_file: str):
    """Load ground truth data from CSV file."""
    gt_data = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data.append(row)
    return gt_data

def generate_llm_answer(question: str, client: OpenAI, model: str = "gpt-4o"):
    """
    Generate an answer to a question using an LLM.
    
    Args:
        question: The question to answer
        client: OpenAI client
        model: Model to use for generation
        
    Returns:
        The generated answer
    """
    try:
        # Remove any problematic characters from the question
        cleaned_question = question.strip()
        
        # Create a simple prompt with just the question
        prompt = f"Answer this question with detailed information: {cleaned_question}"
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Use low temperature for more consistent answers
            max_tokens=500
        )
        
        # Extract the response text
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        time.sleep(2)  # Sleep to handle rate limits
        return ""

def calculate_llm_correctness(hypothesis: str, reference: str, question: str, client: OpenAI, model: str = "gpt-4o"):
    """
    Use an LLM to evaluate the correctness of the hypothesis compared to the reference.
    
    Args:
        hypothesis: The system-generated answer
        reference: The ground truth answer
        question: The original question
        client: OpenAI client
        model: Model to use for evaluation
        
    Returns:
        A score between 0 and 1 representing correctness (1 = fully correct, 0 = incorrect)
    """
    try:
        # Clean inputs to avoid JSON formatting issues
        clean_question = question.strip()
        clean_reference = reference.strip()
        clean_hypothesis = hypothesis.strip()
        
        # Create prompt for the LLM
        prompt = f"""
You are an expert evaluator assessing the correctness of an answer to a question.

Question: {clean_question}

Ground Truth Answer: {clean_reference}

System Answer: {clean_hypothesis}

Evaluate how correct the System Answer is compared to the Ground Truth Answer. Be very critical in your evaluation/analysis.
Give a score from 0 to 1 where:
- 1.0 means the System Answer is fully correct and contains all the information from the Ground Truth
- 0.0 means the System Answer is completely incorrect
- Values between 0 and 1 indicate partial correctness

Output a single line with just the score as a decimal between 0 and 1.
"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
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
        print(f"Error in LLM evaluation: {str(e)}")
        time.sleep(2)  # Sleep to handle rate limits
        return 0.0

def filter_qa_pairs(gt_data, client: OpenAI, llm_model: str = "gpt-4o", threshold: float = 0.5):
    """
    Filter question-answer pairs based on LLM correctness score.
    
    Args:
        gt_data: List of dictionaries containing ground truth data
        client: OpenAI client
        llm_model: Model to use for generation and evaluation
        threshold: Correctness threshold above which pairs are removed
        
    Returns:
        Filtered list of dictionaries
    """
    filtered_data = []
    
    for qa_pair in tqdm(gt_data, desc="Filtering QA pairs"):
        question = qa_pair['question']
        reference_answer = qa_pair['answer']
        
        # Generate an answer using just the question
        llm_answer = generate_llm_answer(question, client, llm_model)
        
        # If we couldn't get an answer, keep the pair
        if not llm_answer:
            filtered_data.append(qa_pair)
            continue
        
        # Evaluate the correctness
        correctness_score = calculate_llm_correctness(
            llm_answer, reference_answer, question, client, llm_model
        )
        
        print(f"Question: {question}")
        print(f"LLM Answer: {llm_answer[:100]}...")
        print(f"Correctness Score: {correctness_score}")
        
        # Keep pairs that the LLM couldn't answer correctly
        if correctness_score <= threshold:
            filtered_data.append(qa_pair)
    
    return filtered_data

def save_filtered_data(filtered_data, output_file):
    """Save filtered data to a CSV file."""
    if not filtered_data:
        print("No data to save!")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=filtered_data[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in filtered_data:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Filter QA pairs based on LLM correctness')
    parser.add_argument('--input', required=True, help='Input ground truth file')
    parser.add_argument('--output', required=True, help='Output filtered file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Correctness threshold (default: 0.5)')
    parser.add_argument('--model', default='gpt-3.5-turbo', 
                        help='OpenAI model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--api-key', help='OpenAI API key (optional if set as environment variable)')
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load ground truth data
    print(f"Loading ground truth data from {args.input}")
    gt_data = load_ground_truth(args.input)
    print(f"Loaded {len(gt_data)} QA pairs")
    
    # Filter pairs
    filtered_data = filter_qa_pairs(gt_data, client, args.model, args.threshold)
    print(f"Filtered to {len(filtered_data)} QA pairs")
    
    # Save filtered data
    save_filtered_data(filtered_data, args.output)
    print(f"Saved filtered data to {args.output}")

if __name__ == "__main__":
    main() 