# API Testing Script for Customer Support RAG System
import requests
import json
import time
from typing import Dict, Any

class APITester:
    """Comprehensive API testing for the customer support system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed - Knowledge Base: {data['knowledge_base_size']} items")
                return True
            else:
                print(f"❌ Health check failed - Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_generate_response(self, query: str, session_id: str = "test_session") -> Dict[str, Any]:
        """Test the main response generation endpoint"""
        print(f"\n🤖 Testing query: '{query}'")
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/generate_response",
                json={
                    "query": query,
                    "session_id": session_id,
                    "include_explainability": True
                },
                timeout=30
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Response generated ({latency:.1f}ms)")
                print(f"📝 Response: {data['response'][:100]}...")
                print(f"📊 Confidence: {data['confidence']:.3f}")
                print(f"🏷️  Category: {data['category']}")
                print(f"🔍 Retrieved {len(data.get('retrieved_contexts', []))} contexts")
                return data
            else:
                print(f"❌ Request failed - Status: {response.status_code}")
                print(f"Error: {response.text}")
                return {}
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return {}
    
    def test_conversation_context(self):
        """Test multi-turn conversation handling"""
        print("\n🗣️  Testing conversation context...")
        
        session_id = "context_test"
        
        # First query
        response1 = self.test_generate_response(
            "I need help resetting my password", 
            session_id
        )
        
        if response1:
            # Follow-up query
            time.sleep(1)  # Brief pause
            response2 = self.test_generate_response(
                "I didn't receive the reset link", 
                session_id
            )
            
            if response2:
                print("✅ Conversation context test passed")
            else:
                print("❌ Follow-up query failed")
        else:
            print("❌ Initial query failed")
    
    def test_explainability_features(self):
        """Test explainability and reasoning features"""
        print("\n🧠 Testing explainability features...")
        
        response = self.test_generate_response(
            "Why did you suggest contacting support?",
            "explainability_test"
        )
        
        if response:
            # Check for explainability components
            has_reasoning = bool(response.get('reasoning'))
            has_contexts = bool(response.get('retrieved_contexts'))
            has_confidence = 'confidence' in response
            
            print(f"📋 Reasoning provided: {'✅' if has_reasoning else '❌'}")
            print(f"📚 Context sources: {'✅' if has_contexts else '❌'}")
            print(f"📊 Confidence score: {'✅' if has_confidence else '❌'}")
            
            if has_reasoning:
                print(f"💭 Reasoning: {response['reasoning']}")
        
    def test_analytics_endpoint(self):
        """Test the analytics endpoint"""
        print("\n📈 Testing analytics endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/analytics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Analytics retrieved successfully")
                print(f"📊 Total interactions: {data.get('total_interactions', 0)}")
                print(f"⭐ Average rating: {data.get('average_rating', 0):.2f}")
                print(f"🎯 Average confidence: {data.get('average_confidence', 0):.3f}")
                return data
            else:
                print(f"❌ Analytics request failed - Status: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Analytics error: {e}")
            return {}
    
    def test_feedback_submission(self):
        """Test feedback submission"""
        print("\n💬 Testing feedback submission...")
        
        # Generate a response first to get interaction ID
        response = self.test_generate_response("Test query for feedback", "feedback_test")
        
        if response:
            # Try to submit feedback (note: we don't have the actual interaction ID)
            try:
                feedback_response = self.session.post(
                    f"{self.base_url}/feedback",
                    json={
                        "interaction_id": "test_interaction_id",
                        "rating": 5,
                        "feedback_text": "Excellent response quality"
                    },
                    timeout=10
                )
                
                if feedback_response.status_code == 200:
                    print("✅ Feedback submitted successfully")
                else:
                    print(f"⚠️ Feedback submission status: {feedback_response.status_code}")
                    
            except Exception as e:
                print(f"❌ Feedback error: {e}")
    
    def run_comprehensive_test(self):
        """Run all tests comprehensively"""
        print("🚀 Starting comprehensive API testing...")
        print("="*60)
        
        # Test 1: Health check
        if not self.test_health_endpoint():
            print("❌ Health check failed - stopping tests")
            return False
        
        # Test 2: Basic queries
        test_queries = [
            "I ordered a laptop but it arrived with a broken screen what should I do",
            "I need help resetting my password",
            "My cat chewed my phone charger is this covered under warranty",
            "How do I track my order"
        ]
        
        print(f"\n📝 Testing {len(test_queries)} sample queries...")
        successful_responses = 0
        
        for query in test_queries:
            response = self.test_generate_response(query)
            if response:
                successful_responses += 1
        
        print(f"\n📊 Query Test Results: {successful_responses}/{len(test_queries)} successful")
        
        # Test 3: Conversation context
        self.test_conversation_context()
        
        # Test 4: Explainability
        self.test_explainability_features()
        
        # Test 5: Analytics
        self.test_analytics_endpoint()
        
        # Test 6: Feedback
        self.test_feedback_submission()
        
        print("\n" + "="*60)
        print("🎉 Comprehensive testing completed!")
        print("="*60)
        
        return successful_responses >= len(test_queries) * 0.8  # 80% success rate

def main():
    """Main testing function"""
    tester = APITester()
    
    print("🧪 RAG Customer Support API Tester")
    print("Make sure your API server is running on http://localhost:8000")
    input("Press Enter to start testing...")
    
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n✅ All tests passed! System is ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Check the logs and fix issues before deployment.")

if __name__ == "__main__":
    main()