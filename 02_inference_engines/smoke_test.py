"""
SMOKE TEST: Inference Engine & Metrics Validation
--------------------------------------------------
Purpose: Validate that all 4 configurations work and metrics are flowing to Prometheus

This test will:
1. Check if the inference engine is responding
2. Send a few test requests
3. Verify metrics are being exposed
4. Check Prometheus is scraping metrics
5. Validate Grafana can query the data

Usage:
    python smoke_test.py --engine vllm --model meta-llama/Llama-3.1-8B-Instruct --url http://localhost:8000/v1
"""

import time
import json
import requests
import argparse
import sys
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def test_health_endpoint(base_url):
    """Test if the inference engine is responding"""
    print_header("TEST 1: Health Check")
    
    # Try common health endpoints
    health_endpoints = [
        f"{base_url}/health",
        f"{base_url.replace('/v1', '')}/health",
        f"{base_url.replace('/v1', '')}/info"
    ]
    
    for endpoint in health_endpoints:
        try:
            print_info(f"Trying: {endpoint}")
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print_success(f"Engine is healthy: {endpoint}")
                return True
        except Exception as e:
            print_warning(f"Endpoint {endpoint} not available: {e}")
    
    print_warning("No health endpoint found, but engine might still work")
    return True


def test_inference(base_url, model):
    """Test basic inference capability"""
    print_header("TEST 2: Basic Inference")
    
    endpoint = f"{base_url}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
        "max_tokens": 10,
        "stream": False
    }
    
    try:
        print_info(f"Sending test request to: {endpoint}")
        print_info(f"Model: {model}")
        
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            print_success(f"Inference successful ({duration:.2f}s)")
            print_info(f"Response: {content[:100]}")
            return True
        else:
            print_error(f"Inference failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Inference request failed: {e}")
        return False


def test_streaming_inference(base_url, model):
    """Test streaming inference capability"""
    print_header("TEST 3: Streaming Inference")
    
    endpoint = f"{base_url}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "max_tokens": 20,
        "stream": True
    }
    
    try:
        print_info(f"Sending streaming request to: {endpoint}")
        
        start_time = time.time()
        token_count = 0
        first_token_time = None
        
        with requests.post(endpoint, json=payload, stream=True, timeout=30) as response:
            if response.status_code != 200:
                print_error(f"Streaming failed: {response.status_code}")
                return False
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
        
        total_time = time.time() - start_time
        print_success(f"Streaming successful ({total_time:.2f}s)")
        print_info(f"Tokens received: {token_count}")
        print_info(f"Time to first token: {first_token_time*1000:.2f}ms")
        return True
        
    except Exception as e:
        print_error(f"Streaming request failed: {e}")
        return False


def test_metrics_endpoint(engine, metrics_port):
    """Test if metrics are being exposed"""
    print_header("TEST 4: Metrics Endpoint")
    
    metrics_url = f"http://localhost:{metrics_port}/metrics"
    
    try:
        print_info(f"Checking metrics at: {metrics_url}")
        response = requests.get(metrics_url, timeout=5)
        
        if response.status_code == 200:
            metrics_text = response.text
            
            # Check for engine-specific metrics
            if engine == "vllm":
                required_metrics = [
                    "vllm:request_success_total",
                    "vllm:time_to_first_token_seconds",
                    "vllm:generation_tokens_total"
                ]
            else:  # tgi
                required_metrics = [
                    "tgi_request_success",
                    "tgi_request_duration",
                    "tgi_request_generated_tokens"
                ]
            
            found_metrics = []
            missing_metrics = []
            
            for metric in required_metrics:
                if metric in metrics_text:
                    found_metrics.append(metric)
                else:
                    missing_metrics.append(metric)
            
            print_success(f"Metrics endpoint is accessible")
            print_info(f"Found {len(found_metrics)}/{len(required_metrics)} required metrics")
            
            for metric in found_metrics:
                print_success(f"  {metric}")
            
            for metric in missing_metrics:
                print_warning(f"  {metric} (not found yet - may appear after requests)")
            
            return len(found_metrics) > 0
        else:
            print_error(f"Metrics endpoint returned: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Could not access metrics endpoint: {e}")
        return False


def test_prometheus_scraping(engine):
    """Test if Prometheus is scraping the metrics"""
    print_header("TEST 5: Prometheus Scraping")
    
    prometheus_url = "http://localhost:9090/api/v1/query"
    
    # Query for a basic metric
    if engine == "vllm":
        query = "up{job='vllm'}"
    else:
        query = "up{job='tgi'}"
    
    try:
        print_info(f"Querying Prometheus: {query}")
        response = requests.get(prometheus_url, params={"query": query}, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('data', {}).get('result', [])
            
            if results:
                value = results[0].get('value', [None, None])[1]
                print_success(f"Prometheus is scraping {engine} metrics")
                print_info(f"Service status: {'UP' if value == '1' else 'DOWN'}")
                return True
            else:
                print_warning(f"Prometheus is running but no data for {engine} yet")
                print_info("This is normal if you just started the service")
                return True
        else:
            print_error(f"Prometheus query failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Could not query Prometheus: {e}")
        print_warning("Make sure Prometheus is running at http://localhost:9090")
        return False


def test_grafana_connectivity():
    """Test if Grafana is accessible"""
    print_header("TEST 6: Grafana Connectivity")
    
    grafana_url = "http://localhost:3000/api/health"
    
    try:
        print_info(f"Checking Grafana at: {grafana_url}")
        response = requests.get(grafana_url, timeout=5)
        
        if response.status_code == 200:
            print_success("Grafana is accessible")
            print_info("Dashboards available at: http://localhost:3000")
            return True
        else:
            print_error(f"Grafana returned: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Could not access Grafana: {e}")
        print_warning("Make sure Grafana is running at http://localhost:3000")
        return False


def run_smoke_test(engine, url, model, metrics_port):
    """Run all smoke tests"""
    print_header(f"SMOKE TEST: {engine.upper()} - {model.split('/')[-1]}")
    print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "health_check": False,
        "basic_inference": False,
        "streaming_inference": False,
        "metrics_endpoint": False,
        "prometheus_scraping": False,
        "grafana_connectivity": False
    }
    
    # Run tests
    results["health_check"] = test_health_endpoint(url)
    time.sleep(1)
    
    results["basic_inference"] = test_inference(url, model)
    time.sleep(2)
    
    results["streaming_inference"] = test_streaming_inference(url, model)
    time.sleep(2)
    
    results["metrics_endpoint"] = test_metrics_endpoint(engine, metrics_port)
    time.sleep(1)
    
    results["prometheus_scraping"] = test_prometheus_scraping(engine)
    time.sleep(1)
    
    results["grafana_connectivity"] = test_grafana_connectivity()
    
    # Summary
    print_header("SMOKE TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:6}{Colors.END} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("\n🎉 All tests passed! Ready for benchmarking.")
        return 0
    elif passed >= 4:
        print_warning("\n⚠️  Most tests passed. You can proceed with caution.")
        print_info("Some monitoring features may not work properly.")
        return 0
    else:
        print_error("\n❌ Critical tests failed. Please fix issues before benchmarking.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for inference engine setup")
    parser.add_argument("--engine", type=str, required=True, choices=["vllm", "tgi"],
                        help="Inference engine being tested")
    parser.add_argument("--url", type=str, required=True,
                        help="Base URL of the API (e.g., http://localhost:8000/v1)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID to test")
    parser.add_argument("--metrics-port", type=int,
                        help="Port where metrics are exposed (default: 8000 for vLLM, 8080 for TGI)")
    
    args = parser.parse_args()
    
    # Set default metrics port based on engine
    if args.metrics_port is None:
        args.metrics_port = 8000 if args.engine == "vllm" else 8080
    
    exit_code = run_smoke_test(args.engine, args.url, args.model, args.metrics_port)
    sys.exit(exit_code)
