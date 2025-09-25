# Simple Evaluation Framework for RAG Customer Support System
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests
from sentence_transformers import SentenceTransformer

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG system"""
    retrieval_accuracy: float
    response_relevance: float
    response_quality: float
    category_accuracy: float
    explainability_score: float
    latency_ms: float
    overall_score: float

class SimpleRAGEvaluator:
    """Simple evaluation system for RAG-based customer support"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test cases with ground truth
        self.test_cases = [
            {
                "query": "I ordered a laptop but it arrived with a broken screen what should I do",
                "expected_category": "technical_issue",
                "expected_keywords": ["replacement", "refund", "contact", "support", "return"],
                "difficulty": "easy"
            },
            {
                "query": "I need help resetting my password",
                "expected_category": "account_access",
                "expected_keywords": ["reset", "password", "email", "link", "login"],
                "difficulty": "easy"
            },
            {
                "query": "I didn't receive the reset link",
                "expected_category": "account_access",
                "expected_keywords": ["spam", "junk", "email", "resend", "manual"],
                "difficulty": "medium"
            },
            {
                "query": "My cat chewed my phone charger is this covered under warranty",
                "expected_category": "warranty",
                "expected_keywords": ["warranty", "not covered", "damage", "accidental", "protection"],
                "difficulty": "hard"
            },
            {
                "query": "How do I track my order and when will it arrive",
                "expected_category": "order_inquiry",
                "expected_keywords": ["track", "order", "account", "tracking", "delivery"],
                "difficulty": "easy"
            },
            {
                "query": "Why did you suggest contacting support",
                "expected_category": "general_inquiry",
                "expected_keywords": ["because", "reason", "explanation", "help", "complex"],
                "difficulty": "hard"
            }
        ]
    
    def evaluate_response_relevance(self, query: str, response: str, expected_keywords: List[str]) -> float:
        """Evaluate if response is relevant to query"""
        if not response or len(response.strip()) < 10:
            return 0.0
            
        response_lower = response.lower()
        
        # Keyword presence score
        keyword_score = sum(1 for keyword in expected_keywords if keyword in response_lower) / len(expected_keywords)
        
        # Semantic similarity
        query_embedding = self.embedding_model.encode([query])
        response_embedding = self.embedding_model.encode([response])
        
        semantic_score = np.dot(query_embedding[0], response_embedding[0]) / (
            np.linalg.norm(query_embedding[0]) * np.linalg.norm(response_embedding[0])
        )
        
        return (keyword_score * 0.6 + max(0, semantic_score) * 0.4)
    
    def evaluate_response_quality(self, response: str) -> float:
        """Evaluate overall quality of response"""
        if not response or len(response.strip()) < 10:
            return 0.0
            
        quality_factors = []
        
        # Length check (optimal range: 50-200 words)
        word_count = len(response.split())
        if 50 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 50:
            length_score = max(0.3, word_count / 50)
        else:
            length_score = max(0.3, 200 / word_count)
        quality_factors.append(length_score)
        
        # Completeness (ends with proper punctuation)
        completeness_score = 1.0 if response.strip().endswith(('.', '!', '?')) else 0.7
        quality_factors.append(completeness_score)
        
        # Helpfulness indicators
        helpful_phrases = ['please', 'you can', 'I recommend', 'contact', 'visit', 'check']
        helpfulness_score = min(1.0, sum(1 for phrase in helpful_phrases if phrase in response.lower()) / 3)
        quality_factors.append(helpfulness_score)
        
        return np.mean(quality_factors)
    
    def run_single_evaluation(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run evaluation on a single test case"""
        import time
        
        print(f"Testing: {test_case['query'][:50]}...")
        
        # Make API request
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_base_url}/generate_response",
                json={
                    "query": test_case["query"],
                    "session_id": "evaluation",
                    "include_explainability": True
                },
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return None
        
        # Extract response components
        ai_response = response_data.get('response', '')
        retrieved_contexts = response_data.get('retrieved_contexts', [])
        predicted_category = response_data.get('query_classification', {}).get('category', '')
        
        # Calculate metrics
        response_relevance = self.evaluate_response_relevance(
            test_case["query"], 
            ai_response, 
            test_case["expected_keywords"]
        )
        response_quality = self.evaluate_response_quality(ai_response)
        category_accuracy = 1.0 if predicted_category == test_case["expected_category"] else 0.0
        retrieval_accuracy = len(retrieved_contexts) * 0.2  # Simple score based on retrieved contexts
        explainability_score = 1.0 if response_data.get('reasoning') else 0.0
        
        # Calculate overall score
        overall_score = (
            retrieval_accuracy * 0.25 +
            response_relevance * 0.25 +
            response_quality * 0.20 +
            category_accuracy * 0.15 +
            explainability_score * 0.15
        )
        
        return {
            "test_case": test_case,
            "response_data": response_data,
            "ai_response": ai_response,
            "metrics": {
                "retrieval_accuracy": retrieval_accuracy,
                "response_relevance": response_relevance,
                "response_quality": response_quality,
                "category_accuracy": category_accuracy,
                "explainability_score": explainability_score,
                "latency_ms": latency_ms,
                "overall_score": overall_score
            }
        }
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test cases"""
        print("🧪 Running comprehensive RAG system evaluation...")
        
        all_results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\nTest case {i}/{len(self.test_cases)}")
            result = self.run_single_evaluation(test_case)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("❌ No successful evaluations!")
            return {}
        
        # Calculate aggregate metrics
        metrics_list = [r["metrics"] for r in all_results]
        avg_metrics = {
            'retrieval_accuracy': np.mean([m["retrieval_accuracy"] for m in metrics_list]),
            'response_relevance': np.mean([m["response_relevance"] for m in metrics_list]),
            'response_quality': np.mean([m["response_quality"] for m in metrics_list]),
            'category_accuracy': np.mean([m["category_accuracy"] for m in metrics_list]),
            'explainability_score': np.mean([m["explainability_score"] for m in metrics_list]),
            'latency_ms': np.mean([m["latency_ms"] for m in metrics_list]),
            'overall_score': np.mean([m["overall_score"] for m in metrics_list])
        }
        
        evaluation_report = {
            'summary': {
                'total_test_cases': len(self.test_cases),
                'successful_evaluations': len(all_results),
                'overall_system_score': avg_metrics['overall_score']
            },
            'average_metrics': avg_metrics,
            'detailed_results': all_results
        }
        
        # Save detailed report
        with open('simple_evaluation_report.json', 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        return evaluation_report

def main():
    """Run the evaluation"""
    evaluator = SimpleRAGEvaluator()
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server not healthy. Make sure main.py is running.")
            return
    except:
        print("❌ Cannot connect to API server. Make sure main.py is running on port 8000.")
        return
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\n" + "="*50)
        print("📊 EVALUATION COMPLETE")
        print("="*50)
        print(f"Overall System Score: {results['average_metrics']['overall_score']:.3f}/1.000")
        print(f"Response Quality: {results['average_metrics']['response_quality']:.3f}")
        print(f"Response Relevance: {results['average_metrics']['response_relevance']:.3f}")
        print(f"Category Accuracy: {results['average_metrics']['category_accuracy']:.3f}")
        print(f"Average Latency: {results['average_metrics']['latency_ms']:.1f}ms")
        print("\n📋 Report saved as: simple_evaluation_report.json")
        print("="*50)
    else:
        print("❌ Evaluation failed - check if the API server is running")

if __name__ == "__main__":
    main()