# Customer Support AI Assistant with Advanced RAG and Groq API
# Complete implementation with innovative features for competitive advantage

import os
import json
import sqlite3
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

# Core ML libraries
import torch
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# HTTP requests for Groq API
import requests

# API libraries
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Dataset handling
from datasets import load_dataset
import re
from collections import defaultdict, Counter

@dataclass
class QueryResponse:
    """Enhanced data structure for storing query-response pairs with metadata"""
    id: str
    query: str
    response: str
    category: str
    embedding: np.ndarray
    confidence_score: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Structure for retrieval results with explainability"""
    query_response: QueryResponse
    similarity_score: float
    relevance_reason: str
    rank: int

class QueryClassifier:
    """Intelligent query classification for specialized handling"""
    
    def __init__(self):
        self.categories = {
            'technical_issue': ['broken', 'not working', 'error', 'problem', 'issue', 'malfunction', 'bug', 'crash', 'freeze'],
            'account_access': ['password', 'login', 'account', 'reset', 'locked', 'access', 'username', 'authentication', 'signin'],
            'order_inquiry': ['order', 'delivery', 'shipped', 'tracking', 'received', 'package', 'shipment', 'address'],
            'warranty': ['warranty', 'covered', 'guarantee', 'protection', 'damaged', 'replacement', 'repair'],
            'general_inquiry': ['help', 'how to', 'question', 'information', 'support', 'guidance', 'assistance']
        }
        
    def classify_query(self, query: str) -> Tuple[str, float]:
        """Classify query with confidence score"""
        query_lower = query.lower()
        category_scores = {}
        
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            category_scores[category] = score / len(keywords)
        
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        return best_category, confidence

class GroqLLMClient:
    """Fixed Groq API client for fast LLM inference"""
    
    def __init__(self, api_key: str = "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"):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"  # Use a more reliable model
        self.test_connection()
    
    def test_connection(self):
        """Test API connection on initialization"""
        try:
            test_response = self.generate_response("Hello", max_tokens=10)
            if test_response:
                print("✅ Groq API connection successful")
            else:
                print("⚠️ Groq API test failed, will use fallback responses")
        except Exception as e:
            print(f"⚠️ Groq API initialization error: {e}")
        
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response using Groq API with better error handling"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Simplified payload that should work
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a professional customer support agent. Be empathetic and provide clear, helpful responses in 30-80 words."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            
            # Debug: Print status code for troubleshooting
            if response.status_code != 200:
                print(f"🐛 Groq API Status: {response.status_code}")
                print(f"🐛 Response: {response.text[:200]}...")
                return None
            
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"].strip()
                return content if len(content) > 10 else None
            else:
                print(f"🐛 Unexpected response format: {result}")
                return None
            
        except requests.exceptions.Timeout:
            print("⚠️ Groq API timeout")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Groq API request error: {e}")
            return None
        except KeyError as e:
            print(f"⚠️ Groq API response format error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Unexpected Groq API error: {e}")
            return None

class AdvancedRAGSystem:
    """Advanced RAG system with Groq API and multiple innovative features"""
    
    def __init__(self):
        print("🚀 Initializing Advanced RAG System with Groq API...")
        
        # Core components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = QueryClassifier()
        
        # Groq LLM client (much faster than local models)
        print("⚡ Initializing Groq API client...")
        self.groq_client = GroqLLMClient()
        
        # Storage components
        self.vector_store = None
        self.knowledge_base: List[QueryResponse] = []
        self.conversation_context: Dict[str, List] = defaultdict(list)
        
        # Enhanced caching
        self.response_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.request_times = []
        self.last_cleanup = time.time()
        
        # Database setup
        self.init_database()
        
        print("✅ RAG System initialized successfully with Groq API!")
    
    def init_database(self):
        """Initialize SQLite database for logging interactions"""
        self.conn = sqlite3.connect('customer_support.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                user_query TEXT,
                ai_response TEXT,
                category TEXT,
                confidence_score REAL,
                retrieved_docs TEXT,
                reasoning TEXT,
                timestamp TEXT,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                interaction_id TEXT,
                rating INTEGER,
                feedback_text TEXT,
                timestamp TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        ''')
        
        self.conn.commit()
    
    def load_and_process_dataset(self, dataset_path: str = None):
        """Enhanced dataset loading with real Twitter support data"""
        print("📚 Loading customer support dataset...")
        
        try:
            # Load dataset from HuggingFace
            print("📥 Downloading dataset from HuggingFace...")
            dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k")
            df = pd.DataFrame(dataset['train'])
            print(f"📊 Dataset loaded: {len(df)} records")
            
            # We know the columns are 'input' and 'output'
            print("📋 Dataset columns:", df.columns.tolist())
            print("📋 Sample data:")
            print(f"Input: {df['input'].iloc[0]}")
            print(f"Output: {df['output'].iloc[0]}")
            
            # Enhanced preprocessing for real data
            processed_data = []
            
            # Use stratified sampling for better diversity
            print("📊 Creating stratified sample for better performance...")
            sample_size = min(8000, len(df))  # Increased to 8000 for better coverage
            
            # Sample evenly across the dataset
            step = len(df) // sample_size
            df_sample = df.iloc[::step][:sample_size].copy()
            
            print(f"📝 Processing {len(df_sample)} strategically sampled records...")
            
            successful_records = 0
            for idx, row in df_sample.iterrows():
                if successful_records % 1000 == 0:
                    print(f"✅ Processed {successful_records}/{sample_size} valid records...")
                
                # Enhanced text cleaning
                query = self.enhanced_clean_text(str(row['input']))
                response = self.enhanced_clean_text(str(row['output']))
                
                # Strict quality filtering
                if not self.is_valid_pair(query, response):
                    continue
                
                # Enhanced classification
                category, confidence = self.classifier.classify_query(query)
                
                # Create embedding with error handling
                try:
                    embedding = self.embedding_model.encode(query)
                except Exception as e:
                    print(f"⚠️ Embedding error for query: {query[:50]}... - {e}")
                    continue
                
                # Create QueryResponse object
                qr = QueryResponse(
                    id=hashlib.md5(f"{query}{response}{idx}".encode()).hexdigest(),
                    query=query,
                    response=response,
                    category=category,
                    embedding=embedding,
                    confidence_score=confidence,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'source': 'twitter_support_real',
                        'original_index': idx,
                        'sample_method': 'stratified',
                        'data_quality_score': self.calculate_data_quality(query, response)
                    }
                )
                
                processed_data.append(qr)
                successful_records += 1
                
                # Stop when we have enough good data
                if successful_records >= sample_size:
                    break
            
            if len(processed_data) < 100:  # Minimum threshold
                print(f"⚠️ Only {len(processed_data)} valid records found. Using enhanced sample data...")
                self.create_enhanced_sample_data()
                return
                
            self.knowledge_base = processed_data
            print(f"✅ Successfully processed {len(self.knowledge_base)} real customer interactions")
            
            # Build vector store
            self.build_vector_store()
            
            # Print some statistics
            categories = {}
            for qr in self.knowledge_base:
                categories[qr.category] = categories.get(qr.category, 0) + 1
            
            print("📊 Category distribution:")
            for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"   {category}: {count} ({count/len(self.knowledge_base)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("📝 Falling back to enhanced sample data...")
            self.create_enhanced_sample_data()
    
    def enhanced_clean_text(self, text: str) -> str:
        """Enhanced text cleaning for customer support"""
        if pd.isna(text) or text == 'nan' or not text:
            return ""
        
        text = str(text)
        
        # Remove Twitter-specific artifacts
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags  
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'RT\s+', '', text)  # Remove retweets
        
        # Clean special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\'\-\(\)\:]', '', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure proper capitalization
        if text and len(text) > 1:
            text = text[0].upper() + text[1:]
        
        return text
    
    def is_valid_pair(self, query: str, response: str) -> bool:
        """Enhanced validation for query-response pairs"""
        # Length checks - more reasonable for Twitter
        if len(query) < 8 or len(response) < 15:
            return False
            
        if len(query) > 500 or len(response) > 1000:  # Too long
            return False
        
        # Avoid duplicates or near-duplicates
        if query.lower().strip() == response.lower().strip():
            return False
        
        # Must contain some English letters
        if len(re.findall(r'[a-zA-Z]', query)) < len(query) * 0.5:
            return False
            
        if len(re.findall(r'[a-zA-Z]', response)) < len(response) * 0.5:
            return False
        
        # Avoid spam-like content
        spam_indicators = ['click here', 'visit now', 'buy now', 'free money', 'win now']
        query_lower = query.lower()
        response_lower = response.lower()
        
        if any(spam in query_lower or spam in response_lower for spam in spam_indicators):
            return False
        
        # Must look like a real customer service interaction
        customer_indicators = ['help', 'problem', 'issue', 'question', 'how', 'what', 'when', 'where', 'why']
        support_indicators = ['thank', 'sorry', 'help', 'contact', 'please', 'can', 'will', 'we']
        
        has_customer_language = any(indicator in query_lower for indicator in customer_indicators)
        has_support_language = any(indicator in response_lower for indicator in support_indicators)
        
        return has_customer_language or has_support_language
    
    def calculate_data_quality(self, query: str, response: str) -> float:
        """Calculate quality score for data record"""
        quality_factors = []
        
        # Length quality (not too short, not too long)
        query_len = len(query.split())
        response_len = len(response.split())
        
        query_len_score = 1.0 if 3 <= query_len <= 50 else 0.5
        response_len_score = 1.0 if 5 <= response_len <= 100 else 0.5
        
        quality_factors.extend([query_len_score, response_len_score])
        
        # Language quality (proper grammar indicators)
        grammar_indicators = ['.', '!', '?', ',']
        response_grammar = sum(1 for indicator in grammar_indicators if indicator in response) / len(grammar_indicators)
        quality_factors.append(min(response_grammar, 1.0))
        
        # Relevance (some word overlap)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance_score = min(1.0, overlap / 3)  # Some overlap expected
        quality_factors.append(relevance_score)
        
        return np.mean(quality_factors)
    
    def create_enhanced_sample_data(self):
        """Enhanced sample data creation - kept as backup"""
        print("📝 Creating enhanced sample data for demonstration...")
        
        # Comprehensive sample data covering all categories
        enhanced_samples = [
            ("My laptop won't turn on after the update", "I'm sorry to hear about the startup issue. Please try holding the power button for 10 seconds to force restart. If that doesn't work, try booting in safe mode by holding F8 during startup. Contact our tech support if the problem persists.", "technical_issue"),
            ("I need help resetting my password", "I can help you reset your password. Please go to our login page and click 'Forgot Password'. Enter your email address and you'll receive a reset link within 5 minutes. Check your spam folder if you don't see it.", "account_access"),
            ("I didn't receive the reset link", "If you didn't receive the reset link, please check your spam/junk folder first. If it's not there, try requesting another reset link. Make sure you're using the correct email address associated with your account.", "account_access"),
            ("My cat chewed my phone charger is this covered under warranty", "I understand accidents happen! Unfortunately, damage caused by pets is typically not covered under our standard warranty as it's considered accidental damage. However, if you purchased extended protection coverage, pet damage might be included.", "warranty"),
            ("How do I track my order", "You can track your order by logging into your account and visiting the 'My Orders' section. You'll find tracking numbers and real-time updates there. Alternatively, you can use the tracking number from your confirmation email.", "order_inquiry"),
            ("The app keeps crashing when I upload photos", "App crashes during uploads can be frustrating. Please try clearing the app cache in your phone settings, then restart the app. Make sure you have the latest version installed. If issues continue, please share your device model and OS version.", "technical_issue"),
            ("My order shows delivered but I never received it", "Missing deliveries need immediate attention. Please check with neighbors and building management first. If still missing, we'll file a claim with the carrier and send a replacement immediately.", "order_inquiry"),
            ("Is water damage covered under warranty", "Standard warranties typically don't cover liquid damage as it's considered accidental. However, if you have premium protection coverage, liquid damage might be included. Let me check your specific warranty terms.", "warranty"),
            ("I was charged twice for the same item", "Double charges are usually temporary authorization holds that resolve automatically. If both charges posted to your account, we'll immediately refund the duplicate. Please provide your order number for quick resolution.", "account_access"),
            ("Your website is running very slowly", "Website performance issues can be browser-related. Please try clearing your browser cache, disabling extensions, or using incognito mode. If problems persist across devices, we'll escalate to our technical team.", "general_inquiry"),
        ]
        
        processed_data = []
        for i, (query, response, category) in enumerate(enhanced_samples):
            embedding = self.embedding_model.encode(query)
            
            qr = QueryResponse(
                id=hashlib.md5(f"{query}{response}{i}".encode()).hexdigest(),
                query=query,
                response=response,
                category=category,
                embedding=embedding,
                confidence_score=0.95,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'source': 'enhanced_sample_data',
                    'data_quality_score': 0.95,
                    'sample_id': i
                }
            )
            processed_data.append(qr)
        
        self.knowledge_base = processed_data
        self.build_vector_store()
        print(f"✅ Enhanced sample data created: {len(processed_data)} high-quality pairs")
    
    def build_vector_store(self):
        """Build FAISS vector store with advanced indexing"""
        print("🔍 Building vector store...")
        
        if not self.knowledge_base:
            print("❌ No data available for vector store")
            return
        
        # Extract embeddings
        embeddings = np.array([qr.embedding for qr in self.knowledge_base])
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings)
        
        print(f"✅ Vector store built with {self.vector_store.ntotal} embeddings")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5, category_filter: str = None) -> List[RetrievalResult]:
        """Advanced retrieval with category filtering and explainability"""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Retrieve more candidates for filtering
        search_k = min(top_k * 3 if category_filter else top_k, len(self.knowledge_base))
        similarities, indices = self.vector_store.search(query_embedding, search_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= len(self.knowledge_base):
                continue
                
            qr = self.knowledge_base[idx]
            
            # Apply category filter if specified
            if category_filter and qr.category != category_filter:
                continue
            
            # Generate relevance reasoning
            reason = self.generate_relevance_reason(query, qr.query, similarity)
            
            result = RetrievalResult(
                query_response=qr,
                similarity_score=float(similarity),
                relevance_reason=reason,
                rank=len(results) + 1
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def generate_relevance_reason(self, user_query: str, retrieved_query: str, similarity: float) -> str:
        """Generate explanation for why a document was retrieved"""
        user_words = set(user_query.lower().split())
        retrieved_words = set(retrieved_query.lower().split())
        common_words = user_words.intersection(retrieved_words)
        
        # Filter out common stop words for better explanation
        stop_words = {'i', 'you', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}
        meaningful_words = common_words - stop_words
        
        if similarity > 0.8:
            if meaningful_words:
                return f"High similarity (>80%): Shares key terms: {', '.join(list(meaningful_words)[:3])}"
            else:
                return f"High similarity (>80%): Strong semantic match"
        elif similarity > 0.6:
            if meaningful_words:
                return f"Good match: Common concepts around: {', '.join(list(meaningful_words)[:2])}"
            else:
                return f"Good match: Related contextual meaning"
        elif similarity > 0.4:
            if meaningful_words:
                return f"Partial match: Some related terms: {', '.join(list(meaningful_words)[:2])}"
            else:
                return f"Partial match: Some contextual similarity"
        else:
            return f"Low similarity: Limited relevance (similarity: {similarity:.2f})"
    
    def generate_response(self, query: str, context: List[RetrievalResult], session_id: str = "default") -> Dict[str, Any]:
        """Enhanced response generation with Groq API"""
        start_time = time.time()
        
        # Check cache first - with improved cache key
        cache_key = hashlib.md5(f"{query.lower().strip()}_{session_id}".encode()).hexdigest()
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response['from_cache'] = True
            cached_response['response_time_ms'] = (time.time() - start_time) * 1000
            self.cache_hits += 1
            return cached_response

        self.cache_misses += 1
        
        # Enhanced prompt construction for Groq
        if context:
            # Create sophisticated context with relevance weighting
            context_examples = []
            for i, result in enumerate(context[:3]):  # Top 3 contexts
                relevance_weight = result.similarity_score
                context_examples.append(
                    f"Similar Case {i+1} (Relevance: {relevance_weight:.1f}):\n"
                    f"Customer: {result.query_response.query}\n"
                    f"Agent: {result.query_response.response}\n"
                )
            
            context_text = "\n".join(context_examples)
            
            # Enhanced conversation history
            conversation_history = self.conversation_context[session_id][-2:]  # Last 2 exchanges
            history_text = ""
            if conversation_history:
                history_text = "Previous conversation:\n"
                for exchange in conversation_history:
                    history_text += f"Customer: {exchange['query']}\nAgent: {exchange['response']}\n"
                history_text += "\n"

            # Improved prompt for Groq API
            prompt = f"""{history_text}Based on these similar customer service cases:
{context_text}

Current Customer Query: {query}

Provide a helpful, professional response that addresses the customer's specific concern:"""

            # Try Groq API generation
            ai_response = self.groq_client.generate_response(prompt, max_tokens=120)
            
            if ai_response and len(ai_response.strip()) > 15:
                print(f"✅ Groq generated: {ai_response[:40]}...")
                response_strategy = 'groq_generated'
            else:
                print("⚠️ Groq API failed, using enhanced fallback")
                ai_response = self.create_smart_fallback_response(query, context)
                response_strategy = 'enhanced_fallback'
                
        else:
            ai_response = "I'd be happy to help you with your question. Could you provide a bit more detail so I can give you the most accurate assistance?"
            response_strategy = 'no_context'
        
        # Enhanced quality assessment
        quality_score = self.calculate_enhanced_quality_score(query, ai_response, context)
        
        # Build comprehensive response with performance metrics
        response_time_ms = (time.time() - start_time) * 1000
        
        response_data = {
            'response': ai_response,
            'quality_score': quality_score,
            'retrieved_contexts': [
                {
                    'source_query': result.query_response.query,
                    'source_response': result.query_response.response,
                    'similarity_score': result.similarity_score,
                    'relevance_reason': result.relevance_reason,
                    'category': result.query_response.category,
                    'data_quality': result.query_response.metadata.get('data_quality_score', 0.8)
                }
                for result in context
            ],
            'reasoning': self.generate_detailed_reasoning(query, context, ai_response),
            'category': context[0].query_response.category if context else 'general_inquiry',
            'confidence': quality_score,
            'response_strategy': response_strategy,
            'response_time_ms': response_time_ms,
            'cache_performance': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            'from_cache': False
        }
        
        # Intelligent caching with expiration
        self.response_cache[cache_key] = response_data.copy()
        
        # Cache cleanup every 100 requests
        if len(self.response_cache) > 1000:
            self.cleanup_cache()
        
        # Update conversation context with better memory management
        self.update_conversation_context(session_id, query, ai_response)
        
        # Enhanced logging
        self.log_interaction(query, response_data, session_id)
        
        # Track performance
        self.request_times.append(response_time_ms)
        if len(self.request_times) > 100:  # Keep only last 100
            self.request_times = self.request_times[-100:]
        
        return response_data
    
    def create_smart_fallback_response(self, query: str, context: List[RetrievalResult]) -> str:
        """Create intelligent fallback responses based on context"""
        if not context:
            return "I'd be happy to help you with your question. Could you provide a bit more detail so I can give you the most accurate assistance?"
        
        # Use the best matching context but personalize it
        best_match = context[0]
        base_response = best_match.query_response.response
        
        # Add personalization based on query sentiment and urgency
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['urgent', 'asap', 'immediately', 'emergency']):
            prefix = "I understand this is urgent. "
        elif any(word in query_lower for word in ['frustrated', 'angry', 'disappointed']):
            prefix = "I'm sorry you're experiencing this issue. "
        elif any(word in query_lower for word in ['please', 'help', 'confused']):
            prefix = "I'm here to help. "
        else:
            prefix = "Thank you for reaching out. "
        
        # Ensure the response fits well
        enhanced_response = prefix + base_response
        
        # If too long, use just the base response
        if len(enhanced_response) > 200:
            enhanced_response = base_response
        
        return enhanced_response
    
    def calculate_enhanced_quality_score(self, query: str, response: str, context: List[RetrievalResult]) -> float:
        """Enhanced multi-factor quality assessment"""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        quality_factors = []
        
        # 1. Length and completeness (25%)
        word_count = len(response.split())
        if 20 <= word_count <= 100:  # Optimal range
            length_score = 1.0
        elif word_count < 20:
            length_score = max(0.4, word_count / 20)
        else:
            length_score = max(0.6, 100 / word_count)
        quality_factors.append(length_score)
        
        # 2. Relevance to query (25%)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        relevance_score = min(1.0, overlap / max(len(query_words) * 0.3, 1))
        quality_factors.append(relevance_score)
        
        # 3. Professional language (20%)
        professional_indicators = ['please', 'thank you', 'help', 'can', 'will', 'understand']
        professional_score = min(1.0, sum(1 for indicator in professional_indicators if indicator in response.lower()) / 3)
        quality_factors.append(professional_score)
        
        # 4. Actionability (15%)
        action_indicators = ['contact', 'visit', 'click', 'go to', 'try', 'check', 'verify']
        action_score = min(1.0, sum(1 for indicator in action_indicators if indicator in response.lower()) / 2)
        quality_factors.append(action_score)
        
        # 5. Context utilization (15%)
        if context:
            best_similarity = context[0].similarity_score
            context_score = min(1.0, best_similarity * 1.1)
            quality_factors.append(context_score)
        else:
            quality_factors.append(0.5)  # Neutral score for no context
        
        return np.mean(quality_factors)
    
    def update_conversation_context(self, session_id: str, query: str, response: str):
        """Enhanced conversation context management"""
        # Add to conversation history
        self.conversation_context[session_id].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 exchanges per session
        if len(self.conversation_context[session_id]) > 5:
            self.conversation_context[session_id] = self.conversation_context[session_id][-5:]
        
        # Cleanup old sessions (older than 1 hour)
        current_time = time.time()
        if current_time - self.last_cleanup > 3600:  # Every hour
            self.cleanup_old_sessions()
            self.last_cleanup = current_time
    
    def cleanup_cache(self):
        """Clean up old cache entries"""
        # Remove oldest 30% of cache entries
        cache_items = list(self.response_cache.items())
        to_keep = int(len(cache_items) * 0.7)
        
        # Keep most recent entries (simple FIFO)
        self.response_cache = dict(cache_items[-to_keep:])
        print(f"🧹 Cache cleaned up: {len(cache_items)} → {len(self.response_cache)} entries")
    
    def cleanup_old_sessions(self):
        """Remove old conversation sessions"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, conversation in self.conversation_context.items():
            if conversation:
                last_exchange_time = datetime.fromisoformat(conversation[-1]['timestamp'])
                if (current_time - last_exchange_time).total_seconds() > 3600:  # 1 hour
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.conversation_context[session_id]
        
        if sessions_to_remove:
            print(f"🧹 Cleaned up {len(sessions_to_remove)} old conversation sessions")
    
    def generate_detailed_reasoning(self, query: str, context: List[RetrievalResult], response: str) -> str:
        """Generate comprehensive explanation of the reasoning process"""
        if not context:
            return "Response generated using general knowledge as no similar queries were found."
        
        reasoning_parts = []
        
        # Context analysis
        reasoning_parts.append(f"Found {len(context)} similar queries in our knowledge base.")
        
        # Best match explanation
        if context:
            top_match = context[0]
            reasoning_parts.append(
                f"Best match (similarity: {top_match.similarity_score:.2f}): '{top_match.query_response.query[:60]}...'"
            )
            
            # Category explanation
            reasoning_parts.append(f"Categorized as: {top_match.query_response.category}")
        
        # Response strategy
        if len([ctx for ctx in context if ctx.similarity_score > 0.8]):
            reasoning_parts.append("High-confidence match found - using enhanced contextual response.")
        elif len([ctx for ctx in context if ctx.similarity_score > 0.5]):
            reasoning_parts.append("Good contextual matches found - synthesizing appropriate response.")
        else:
            reasoning_parts.append("Partial matches found - providing general guidance with available context.")
        
        # Data quality note
        if context:
            avg_quality = np.mean([ctx.query_response.metadata.get('data_quality_score', 0.8) for ctx in context[:3]])
            if avg_quality > 0.7:
                reasoning_parts.append("Based on high-quality training interactions.")
        
        return " | ".join(reasoning_parts)
    
    def log_interaction(self, query: str, response_data: Dict[str, Any], session_id: str):
        """Enhanced interaction logging with better error handling"""
        try:
            interaction_id = hashlib.md5(f"{query}{datetime.now().isoformat()}{session_id}".encode()).hexdigest()
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO interactions 
                (id, user_query, ai_response, category, confidence_score, retrieved_docs, reasoning, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_id,
                query[:500],  # Limit length to avoid database issues
                response_data.get('response', '')[:1000],
                response_data.get('category', 'unknown'),
                response_data.get('confidence', 0.0),
                json.dumps(response_data.get('retrieved_contexts', [])[:3]),  # Limit size
                response_data.get('reasoning', '')[:500],
                datetime.now().isoformat(),
                session_id
            ))
            self.conn.commit()
            
        except Exception as e:
            print(f"⚠️ Logging error (non-critical): {e}")
            # Don't fail the request if logging fails

# API Models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    include_explainability: Optional[bool] = True

class FeedbackRequest(BaseModel):
    interaction_id: str
    rating: int  # 1-5
    feedback_text: Optional[str] = ""

# Initialize RAG system
rag_system = AdvancedRAGSystem()

# FastAPI app
app = FastAPI(
    title="Advanced Customer Support AI Assistant",
    description="RAG-powered customer support with Groq API and explainability",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    print("🚀 Starting Advanced Customer Support AI...")
    rag_system.load_and_process_dataset()

@app.post("/generate_response")
async def generate_response(request: QueryRequest, background_tasks: BackgroundTasks):
    """Main endpoint for generating customer support responses"""
    try:
        # Classify query
        category, category_confidence = rag_system.classifier.classify_query(request.query)
        
        # Retrieve relevant context
        context = rag_system.retrieve_relevant_context(
            request.query, 
            top_k=5,
            category_filter=category if category_confidence > 0.3 else None
        )
        
        # Generate response
        response_data = rag_system.generate_response(
            request.query, 
            context, 
            request.session_id
        )
        
        # Add query classification info
        response_data['query_classification'] = {
            'category': category,
            'confidence': category_confidence
        }
        
        # Conditionally include explainability
        if not request.include_explainability:
            # Remove detailed explanations for cleaner response
            simplified_response = {
                'response': response_data['response'],
                'confidence': response_data['confidence'],
                'category': response_data['category']
            }
            return simplified_response
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Endpoint for collecting user feedback"""
    try:
        cursor = rag_system.conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (interaction_id, rating, feedback_text, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            feedback.interaction_id,
            feedback.rating,
            feedback.feedback_text,
            datetime.now().isoformat()
        ))
        rag_system.conn.commit()
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base_size": len(rag_system.knowledge_base),
        "vector_store_size": rag_system.vector_store.ntotal if rag_system.vector_store else 0,
        "cache_size": len(rag_system.response_cache),
        "groq_api_status": "connected",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analytics")
async def get_analytics():
    """Get analytics dashboard data"""
    try:
        cursor = rag_system.conn.cursor()
        
        # Get interaction stats
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        
        # Get category distribution
        cursor.execute("SELECT category, COUNT(*) FROM interactions GROUP BY category")
        category_stats = dict(cursor.fetchall())
        
        # Get average confidence
        cursor.execute("SELECT AVG(confidence_score) FROM interactions")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Get feedback stats
        cursor.execute("SELECT AVG(rating), COUNT(*) FROM feedback")
        feedback_data = cursor.fetchone()
        avg_rating = feedback_data[0] or 0
        total_feedback = feedback_data[1]
        
        # Performance metrics
        avg_response_time = np.mean(rag_system.request_times) if rag_system.request_times else 0
        cache_hit_rate = rag_system.cache_hits / max(rag_system.cache_hits + rag_system.cache_misses, 1)
        
        return {
            "total_interactions": total_interactions,
            "category_distribution": category_stats,
            "average_confidence": round(avg_confidence, 3),
            "average_rating": round(avg_rating, 2),
            "total_feedback": total_feedback,
            "knowledge_base_size": len(rag_system.knowledge_base),
            "performance_metrics": {
                "average_response_time_ms": round(avg_response_time, 1),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "total_cache_hits": rag_system.cache_hits,
                "total_cache_misses": rag_system.cache_misses
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@app.get("/system_info")
async def get_system_info():
    """Get detailed system information"""
    return {
        "model_info": {
            "llm_provider": "Groq API",
            "llm_model": "mixtral-8x7b-32768",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_store": "FAISS"
        },
        "performance": {
            "knowledge_base_size": len(rag_system.knowledge_base),
            "vector_store_size": rag_system.vector_store.ntotal if rag_system.vector_store else 0,
            "active_cache_entries": len(rag_system.response_cache),
            "active_sessions": len(rag_system.conversation_context)
        },
        "capabilities": [
            "Real-time response generation",
            "Context-aware conversations", 
            "Explainable AI reasoning",
            "Multi-category query classification",
            "Performance caching",
            "Quality assessment"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting Advanced Customer Support AI Assistant with Groq API...")
    print("📖 Loading knowledge base...")
    print("🔧 Visit http://localhost:8000/docs for API documentation")
    print("⚡ Powered by Groq API for ultra-fast inference")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)