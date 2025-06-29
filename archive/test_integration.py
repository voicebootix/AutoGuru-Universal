#!/usr/bin/env python3
"""
AutoGuru Universal - Integration Test Suite

This script tests the complete AutoGuru Universal workflow including:
- API health and connectivity
- Content analysis for multiple business types
- Persona generation
- Viral content creation
- Database operations
- Celery task processing
- All 10 modules working together

Tests cover fitness, business consulting, creative, education, and other niches.
"""

import asyncio
import httpx
import json
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoGuruIntegrationTester:
    """Integration tester for AutoGuru Universal"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.test_results = []
        self.auth_token = None
        
    async def test_api_health(self) -> bool:
        """Test API health and connectivity"""
        print("🏥 Testing API health...")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ API is healthy - Status: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"❌ API health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    async def test_database_connection(self) -> bool:
        """Test database connectivity"""
        print("🗄️ Testing database connection...")
        
        try:
            # Import and test database connection
            from backend.database.connection import check_database_connection
            
            is_connected = await check_database_connection()
            if is_connected:
                print("✅ Database connection successful")
                return True
            else:
                print("❌ Database connection failed")
                return False
                
        except Exception as e:
            print(f"❌ Database test error: {e}")
            return False
    
    async def test_content_analysis(self, test_cases: Dict[str, Dict]) -> Dict[str, Any]:
        """Test content analysis for multiple business types"""
        print("\n📊 Testing content analysis for multiple business types...")
        
        results = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for business_type, test_data in test_cases.items():
                print(f"\n  🎯 Testing {business_type.replace('_', ' ').title()}...")
                
                try:
                    # Prepare request payload
                    payload = {
                        "content": test_data["content"],
                        "context": test_data.get("context", {}),
                        "platforms": test_data.get("platforms", ["instagram", "linkedin", "tiktok"])
                    }
                    
                    # Make API request
                    response = await client.post(
                        f"{self.api_url}/analyze",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results[business_type] = {
                            "success": True,
                            "data": result,
                            "analysis_id": result.get("analysis_id")
                        }
                        
                        # Display analysis results
                        print(f"    ✅ Analysis completed")
                        
                        # Check business niche detection
                        if 'business_niche' in result:
                            niche_data = result['business_niche']
                            niche_type = niche_data.get('niche_type', 'unknown')
                            confidence = niche_data.get('confidence_score', 0)
                            print(f"    ✅ Niche: {niche_type} ({confidence*100:.1f}% confidence)")
                            
                            # Validate expected niche
                            expected_niche = test_data.get("expected_niche")
                            if expected_niche and niche_type == expected_niche:
                                print(f"    ✅ Niche detection accurate")
                            else:
                                print(f"    ⚠️  Expected: {expected_niche}, Got: {niche_type}")
                        
                        # Check target audience analysis
                        if 'target_audience' in result:
                            audience = result['target_audience']
                            print(f"    ✅ Target audience identified")
                            if isinstance(audience, dict) and audience.get('demographics'):
                                print(f"    ✅ Demographics analyzed")
                        
                        # Check brand voice analysis
                        if 'brand_voice' in result:
                            voice = result['brand_voice']
                            print(f"    ✅ Brand voice analyzed")
                            if isinstance(voice, dict) and voice.get('tone'):
                                print(f"    ✅ Voice tone identified")
                        
                        # Check viral potential
                        if 'viral_potential' in result:
                            viral = result['viral_potential']
                            print(f"    ✅ Viral potential calculated")
                            if isinstance(viral, dict):
                                for platform, score in viral.items():
                                    print(f"    ✅ {platform}: {score*100:.1f}% viral potential")
                        
                    else:
                        error_msg = f"Analysis failed: {response.status_code}"
                        if response.text:
                            try:
                                error_data = response.json()
                                error_msg += f" - {error_data.get('detail', response.text)}"
                            except:
                                error_msg += f" - {response.text}"
                        
                        print(f"    ❌ {error_msg}")
                        results[business_type] = {
                            "success": False,
                            "error": error_msg
                        }
                        
                except Exception as e:
                    error_msg = f"Error testing {business_type}: {str(e)}"
                    print(f"    ❌ {error_msg}")
                    results[business_type] = {
                        "success": False,
                        "error": error_msg
                    }
        
        return results
    
    async def test_persona_generation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test persona generation based on analysis results"""
        print("\n👥 Testing persona generation...")
        
        results = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for business_type, analysis_result in analysis_results.items():
                if not analysis_result.get("success"):
                    print(f"  ⏭️  Skipping {business_type} - analysis failed")
                    continue
                
                print(f"  🎭 Generating persona for {business_type.replace('_', ' ').title()}...")
                
                try:
                    # Prepare persona generation request
                    payload = {
                        "business_description": f"Test business for {business_type}",
                        "target_market": "Test target market",
                        "goals": ["increase engagement", "grow audience", "drive conversions"]
                    }
                    
                    response = await client.post(
                        f"{self.api_url}/generate-persona",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results[business_type] = {
                            "success": True,
                            "data": result,
                            "persona_id": result.get("persona_id")
                        }
                        
                        print(f"    ✅ Persona generated successfully")
                        
                        # Display persona details
                        if 'audience_segments' in result:
                            segments = result['audience_segments']
                            print(f"    ✅ {len(segments)} audience segments identified")
                        
                        if 'personality_traits' in result:
                            traits = result['personality_traits']
                            print(f"    ✅ Personality traits analyzed")
                        
                    else:
                        error_msg = f"Persona generation failed: {response.status_code}"
                        if response.text:
                            try:
                                error_data = response.json()
                                error_msg += f" - {error_data.get('detail', response.text)}"
                            except:
                                error_msg += f" - {response.text}"
                        
                        print(f"    ❌ {error_msg}")
                        results[business_type] = {
                            "success": False,
                            "error": error_msg
                        }
                        
                except Exception as e:
                    error_msg = f"Error generating persona for {business_type}: {str(e)}"
                    print(f"    ❌ {error_msg}")
                    results[business_type] = {
                        "success": False,
                        "error": error_msg
                    }
        
        return results
    
    async def test_viral_content_creation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test viral content creation"""
        print("\n🚀 Testing viral content creation...")
        
        results = {}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for business_type, analysis_result in analysis_results.items():
                if not analysis_result.get("success"):
                    print(f"  ⏭️  Skipping {business_type} - analysis failed")
                    continue
                
                print(f"  ✨ Creating viral content for {business_type.replace('_', ' ').title()}...")
                
                try:
                    # Get analysis data
                    analysis_data = analysis_result.get("data", {})
                    
                    # Prepare viral content request
                    payload = {
                        "topic": f"Amazing {business_type.replace('_', ' ')} insights",
                        "business_niche": analysis_data.get("business_niche", {}),
                        "target_audience": analysis_data.get("target_audience", {}),
                        "platforms": ["instagram", "linkedin", "tiktok"],
                        "content_type": "post"
                    }
                    
                    response = await client.post(
                        f"{self.api_url}/create-viral-content",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results[business_type] = {
                            "success": True,
                            "data": result,
                            "content_count": len(result) if isinstance(result, list) else 1
                        }
                        
                        print(f"    ✅ Viral content created successfully")
                        
                        # Display content details
                        if isinstance(result, list):
                            print(f"    ✅ Generated {len(result)} content pieces")
                            for i, content in enumerate(result[:3]):  # Show first 3
                                platform = content.get("platform", "unknown")
                                format_type = content.get("content_format", "unknown")
                                print(f"    ✅ {platform} - {format_type}")
                        else:
                            platform = result.get("platform", "unknown")
                            format_type = result.get("content_format", "unknown")
                            print(f"    ✅ {platform} - {format_type}")
                        
                    else:
                        error_msg = f"Viral content creation failed: {response.status_code}"
                        if response.text:
                            try:
                                error_data = response.json()
                                error_msg += f" - {error_data.get('detail', response.text)}"
                            except:
                                error_msg += f" - {response.text}"
                        
                        print(f"    ❌ {error_msg}")
                        results[business_type] = {
                            "success": False,
                            "error": error_msg
                        }
                        
                except Exception as e:
                    error_msg = f"Error creating viral content for {business_type}: {str(e)}"
                    print(f"    ❌ {error_msg}")
                    results[business_type] = {
                        "success": False,
                        "error": error_msg
                    }
        
        return results
    
    async def test_celery_tasks(self) -> bool:
        """Test Celery task processing"""
        print("\n⚙️ Testing Celery task processing...")
        
        try:
            # Import Celery app
            from backend.tasks.content_generation import app as celery_app
            
            # Test task inspection
            inspector = celery_app.control.inspect()
            active_workers = inspector.active()
            
            if active_workers:
                print("✅ Celery workers are active")
                for worker, tasks in active_workers.items():
                    print(f"    ✅ Worker {worker}: {len(tasks)} active tasks")
                return True
            else:
                print("⚠️  No active Celery workers found")
                return False
                
        except Exception as e:
            print(f"❌ Celery test error: {e}")
            return False
    
    async def test_database_operations(self) -> bool:
        """Test database operations"""
        print("\n💾 Testing database operations...")
        
        try:
            from backend.database.connection import get_db_context
            from backend.models.content_models import ContentAnalysis
            
            async with get_db_context() as session:
                # Test basic database operations
                result = await session.execute("SELECT 1 as test")
                test_value = result.scalar()
                
                if test_value == 1:
                    print("✅ Database query successful")
                    
                    # Test model operations
                    try:
                        # Test creating a sample analysis record
                        sample_analysis = ContentAnalysis(
                            client_id="test_client",
                            business_niche="test_niche",
                            confidence_score=0.95,
                            target_audience={"test": "data"},
                            brand_voice={"test": "voice"},
                            viral_potential={"instagram": 0.8},
                            key_themes=["test", "theme"],
                            recommendations=["test", "recommendation"],
                            metadata={"test": "metadata"}
                        )
                        
                        session.add(sample_analysis)
                        await session.commit()
                        print("✅ Database model operations successful")
                        
                        # Clean up test data
                        await session.delete(sample_analysis)
                        await session.commit()
                        print("✅ Database cleanup successful")
                        
                        return True
                        
                    except Exception as e:
                        print(f"❌ Database model operations failed: {e}")
                        return False
                else:
                    print("❌ Database query failed")
                    return False
                    
        except Exception as e:
            print(f"❌ Database operations test error: {e}")
            return False
    
    def generate_test_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get("success", False))
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful: {successful_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "  Success Rate: 0%")
        
        # Detailed results
        print(f"\n📝 DETAILED RESULTS:")
        for test_name, result in results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"  {status} {test_name}")
            
            if not result.get("success", False):
                error = result.get("error", "Unknown error")
                print(f"      Error: {error}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if failed_tests == 0:
            print("  🎉 All tests passed! AutoGuru Universal is working perfectly.")
        else:
            print("  🔧 Some tests failed. Please check the error messages above.")
            print("  🔧 Ensure all services are running: Docker, FastAPI, Celery")
            print("  🔧 Check environment variables and API keys")
        
        print(f"\n🚀 NEXT STEPS:")
        print("  1. Open frontend/index.html in your browser")
        print("  2. Test the web interface with different business content")
        print("  3. Check the API docs at http://localhost:8000/docs")
        print("  4. Explore the database with pgAdmin at http://localhost:5050")
        print("  5. Monitor Celery tasks and Redis at http://localhost:8081")


async def test_autoguru_universal():
    """Main integration test function"""
    print("🧪 AutoGuru Universal Integration Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize tester
    tester = AutoGuruIntegrationTester()
    
    # Test cases for different business types
    test_cases = {
        "fitness_studio": {
            "content": "Transform your body with our revolutionary 8-week HIIT program! Our certified trainers combine cutting-edge fitness science with personalized nutrition plans. Join thousands who've achieved their dream physique through our proven methodology.",
            "expected_niche": "fitness_wellness",
            "context": {
                "website": "fitnessguru.com",
                "existing_audience": "health enthusiasts",
                "business_type": "fitness_studio"
            },
            "platforms": ["instagram", "youtube", "tiktok"]
        },
        "business_consultant": {
            "content": "Scale your consulting business to 7 figures with our proven frameworks and systems. I help ambitious consultants build predictable revenue streams and automate client acquisition. Learn the exact strategies that took me from $0 to $2M in revenue.",
            "expected_niche": "business_consulting",
            "context": {
                "website": "consultingpro.com",
                "existing_audience": "business owners",
                "business_type": "consulting"
            },
            "platforms": ["linkedin", "twitter", "youtube"]
        },
        "photographer": {
            "content": "Capture life's precious moments with artistic vision and technical excellence. Specializing in wedding photography, portraits, and lifestyle shoots with stunning composition. Every image tells a story - let me tell yours.",
            "expected_niche": "creative",
            "context": {
                "website": "artisticlens.com",
                "existing_audience": "brides and families",
                "business_type": "photography"
            },
            "platforms": ["instagram", "pinterest", "facebook"]
        },
        "online_educator": {
            "content": "Master data science and machine learning with our comprehensive online bootcamp. Learn Python, R, SQL, and cutting-edge AI techniques from industry experts. Join 10,000+ students who've transformed their careers.",
            "expected_niche": "education",
            "context": {
                "website": "datascienceacademy.com",
                "existing_audience": "professionals",
                "business_type": "online_education"
            },
            "platforms": ["youtube", "linkedin", "twitter"]
        },
        "ecommerce_store": {
            "content": "Discover sustainable fashion that doesn't compromise on style. Our ethically-made clothing line uses eco-friendly materials and fair-trade practices. Join the movement towards conscious consumerism.",
            "expected_niche": "ecommerce",
            "context": {
                "website": "ecofashion.com",
                "existing_audience": "conscious consumers",
                "business_type": "ecommerce"
            },
            "platforms": ["instagram", "tiktok", "facebook"]
        }
    }
    
    # Test results storage
    test_results = {}
    
    # Test 1: API Health
    test_results["API Health"] = {
        "success": await tester.test_api_health()
    }
    
    # Test 2: Database Connection
    test_results["Database Connection"] = {
        "success": await tester.test_database_connection()
    }
    
    # Test 3: Content Analysis
    analysis_results = await tester.test_content_analysis(test_cases)
    test_results["Content Analysis"] = {
        "success": any(result.get("success", False) for result in analysis_results.values()),
        "details": analysis_results
    }
    
    # Test 4: Persona Generation
    persona_results = await tester.test_persona_generation(analysis_results)
    test_results["Persona Generation"] = {
        "success": any(result.get("success", False) for result in persona_results.values()),
        "details": persona_results
    }
    
    # Test 5: Viral Content Creation
    viral_results = await tester.test_viral_content_creation(analysis_results)
    test_results["Viral Content Creation"] = {
        "success": any(result.get("success", False) for result in viral_results.values()),
        "details": viral_results
    }
    
    # Test 6: Celery Tasks
    test_results["Celery Tasks"] = {
        "success": await tester.test_celery_tasks()
    }
    
    # Test 7: Database Operations
    test_results["Database Operations"] = {
        "success": await tester.test_database_operations()
    }
    
    # Generate comprehensive report
    tester.generate_test_report(test_results)
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_autoguru_universal())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        sys.exit(1) 