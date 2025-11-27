import time
import re
import argparse
import numpy as np
from datasets import load_dataset
from openai import OpenAI

def extract_answer(text):
    """
    Extracts the numerical answer from GSM8K ground truth or model output.
    GSM8K ground truth format: ".... #### 42"
    """
    # Look for the last number after ####
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(',', '')
    
    # If no ####, look for the last number in the text (heuristic for model output)
    # This is a simple heuristic; more robust extraction might be needed for production
    matches = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(',', '')
    return None

def evaluate(engine_url, model_name, num_samples=20):
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Take a subset
    subset = dataset.select(range(num_samples))
    
    client = OpenAI(base_url=engine_url, api_key="EMPTY")
    
    results = []
    print(f"\nStarting evaluation on {num_samples} samples...")
    print(f"Engine: {engine_url}")
    print(f"Model: {model_name}")
    print("-" * 60)
    
    correct_count = 0
    total_tokens = 0
    total_duration = 0
    
    for i, item in enumerate(subset):
        question = item['question']
        ground_truth_str = item['answer']
        ground_truth = extract_answer(ground_truth_str)
        
        prompt = f"Question: {question}\nAnswer: Let's think step by step."
        
        start_time = time.time()
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=512,
                temperature=0.0  # Greedy decoding for reproducibility
            )
            end_time = time.time()
            
            generated_text = response.choices[0].text
            duration = end_time - start_time
            tokens = response.usage.completion_tokens
            
            model_answer = extract_answer(generated_text)
            
            is_correct = False
            if model_answer and ground_truth:
                try:
                    if float(model_answer) == float(ground_truth):
                        is_correct = True
                except ValueError:
                    pass
            
            if is_correct:
                correct_count += 1
                
            total_tokens += tokens
            total_duration += duration
            
            throughput = tokens / duration if duration > 0 else 0
            
            results.append({
                "id": i,
                "correct": is_correct,
                "tokens": tokens,
                "duration": duration,
                "throughput": throughput
            })
            
            status = "✅" if is_correct else "❌"
            print(f"Sample {i+1}/{num_samples}: {status} | T/s: {throughput:.2f} | GT: {ground_truth} | Pred: {model_answer}")
            
        except Exception as e:
            print(f"Error on sample {i+1}: {e}")

    accuracy = (correct_count / num_samples) * 100
    avg_throughput = np.mean([r['throughput'] for r in results]) if results else 0
    avg_latency = np.mean([r['duration'] for r in results]) if results else 0
    
    print("-" * 60)
    print(f"RESULTS SUMMARY")
    print("-" * 60)
    print(f"Accuracy:       {accuracy:.2f}% ({correct_count}/{num_samples})")
    print(f"Avg Throughput: {avg_throughput:.2f} tokens/sec")
    print(f"Avg Latency:    {avg_latency:.2f} sec/request")
    print(f"Efficiency Score (Acc * T/s): {accuracy * avg_throughput:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Inference Engine Accuracy vs Throughput")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="Inference engine API URL")
    parser.add_argument("--model", type=str, required=True, help="Model name to query")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    evaluate(args.url, args.model, args.samples)
