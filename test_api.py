#!/usr/bin/env python3
"""
Test script for HackRx Multimodal RAG API
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

class HackRxAPITester:
    def __init__(self, base_url: str = "http://localhost:8000", bearer_token: str = "test_token"):
        self.base_url = base_url
        self.bearer_token = bearer_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }
    
    async def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint"""
        print("🔍 Testing health check...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/hackrx/health", headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health check passed")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Services: {data.get('services')}")
                    return data
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return {"error": f"HTTP {response.status}"}
    
    async def test_main_endpoint(self, test_document_url: str, test_questions: list) -> Dict[str, Any]:
        """Test the main /hackrx/run endpoint"""
        print("\n🚀 Testing main endpoint...")
        
        payload = {
            "documents": test_document_url,
            "questions": test_questions
        }
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/hackrx/run", 
                headers=self.headers,
                json=payload
            ) as response:
                end_time = time.time()
                processing_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    print("✅ Main endpoint test passed")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Document type: {data.get('document_type')}")
                    print(f"   Extraction method: {data.get('extraction_method')}")
                    print(f"   Number of answers: {len(data.get('answers', []))}")
                    
                    # Print first answer as example
                    if data.get('answers'):
                        print(f"   Sample answer: {data['answers'][0][:100]}...")
                    
                    return data
                else:
                    error_text = await response.text()
                    print(f"❌ Main endpoint test failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
    
    async def test_invalid_token(self) -> Dict[str, Any]:
        """Test authentication with invalid token"""
        print("\n🔒 Testing invalid token...")
        
        invalid_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer invalid_token"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/hackrx/run",
                headers=invalid_headers,
                json={"documents": "test", "questions": ["test"]}
            ) as response:
                if response.status == 401:
                    print("✅ Invalid token test passed (correctly rejected)")
                    return {"status": "correctly_rejected"}
                else:
                    print(f"❌ Invalid token test failed: {response.status}")
                    return {"error": f"Expected 401, got {response.status}"}
    
    async def test_missing_token(self) -> Dict[str, Any]:
        """Test request without token"""
        print("\n🔒 Testing missing token...")
        
        headers_without_token = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/hackrx/run",
                headers=headers_without_token,
                json={"documents": "test", "questions": ["test"]}
            ) as response:
                if response.status == 403:
                    print("✅ Missing token test passed (correctly rejected)")
                    return {"status": "correctly_rejected"}
                else:
                    print(f"❌ Missing token test failed: {response.status}")
                    return {"error": f"Expected 403, got {response.status}"}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests"""
        print("🧪 Starting HackRx API Tests")
        print("=" * 50)
        
        results = {}
        
        # Test health check
        results["health_check"] = await self.test_health_check()
        
        # Test authentication
        results["invalid_token"] = await self.test_invalid_token()
        results["missing_token"] = await self.test_missing_token()
        
        # Test main endpoint with sample data
        test_document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        test_questions = [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
        
        results["main_endpoint"] = await self.test_main_endpoint(test_document_url, test_questions)
        
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        for test_name, result in results.items():
            if "error" in result:
                print(f"❌ {test_name}: FAILED - {result['error']}")
            else:
                print(f"✅ {test_name}: PASSED")
        
        return results

async def main():
    """Main test function"""
    # Configuration
    base_url = "http://localhost:8000"  # Change to https://localhost:8000 for HTTPS
    bearer_token = "your_bearer_token_here"  # Replace with your actual token
    
    # Create tester instance
    tester = HackRxAPITester(base_url, bearer_token)
    
    # Run tests
    results = await tester.run_all_tests()
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Test results saved to test_results.json")

if __name__ == "__main__":
    asyncio.run(main())