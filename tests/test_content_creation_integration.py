"""
Test Content Creation Integration

Tests for the complete GROUP 4: Content Creation Enhancement modules.
"""

import pytest
import asyncio
from datetime import datetime

from backend.content import (
    ContentType,
    CreativeStyle,
    QualityLevel,
    CreativeRequest,
    CreativeAsset,
    BrandAssetManager,
    CreativePerformanceAnalyzer
)


class TestContentCreationIntegration:
    """Test integration of all content creation modules"""
    
    @pytest.mark.asyncio
    async def test_brand_asset_workflow(self):
        """Test complete brand asset management workflow"""
        
        request = CreativeRequest(
            request_id="test_brand_001",
            client_id="test_client_content",
            content_type=ContentType.BRAND_ASSET,
            business_niche="creative_arts",
            target_audience={"age_range": "25-40", "interests": ["design", "creativity"]},
            creative_brief="Create comprehensive brand system for design studio",
            style_preferences=[CreativeStyle.ARTISTIC, CreativeStyle.MODERN],
            quality_level=QualityLevel.PREMIUM,
            platform_requirements={"instagram": "all", "behance": "all"},
            brand_guidelines={"artistic_direction": "minimalist", "color_preference": "monochromatic"}
        )
        
        brand_manager = BrandAssetManager("test_client_content")
        asset = await brand_manager.create_content(request)
        
        assert asset.content_type == ContentType.BRAND_ASSET
        assert asset.quality_score > 0.8
        assert "instagram" in asset.platform_optimized_versions
        
        # Test platform optimization
        optimized = await brand_manager.optimize_for_platform(asset, "instagram")
        assert "instagram" in optimized.platform_optimized_versions
        
        # Test performance analysis
        performance = await brand_manager.analyze_performance(asset)
        assert "brand_consistency" in performance
        assert performance["compliance_score"]["overall_compliance"] > 0.8

    @pytest.mark.asyncio
    async def test_performance_analysis_workflow(self):
        """Test complete performance analysis workflow"""
        
        request = CreativeRequest(
            request_id="test_analysis_001",
            client_id="test_client_content",
            content_type=ContentType.PERFORMANCE_ANALYSIS,
            business_niche="fitness",
            target_audience={"age_range": "25-35", "interests": ["fitness", "health"]},
            creative_brief="Analyze performance of all fitness content across platforms",
            style_preferences=[CreativeStyle.PROFESSIONAL],
            quality_level=QualityLevel.HIGH,
            platform_requirements={"instagram": "all", "tiktok": "all", "youtube": "all"},
            brand_guidelines={}
        )
        
        analyzer = CreativePerformanceAnalyzer("test_client_content")
        asset = await analyzer.create_content(request)
        
        assert asset.content_type == ContentType.PERFORMANCE_ANALYSIS
        assert asset.quality_score > 0.8
        assert len(asset.platform_optimized_versions) >= 3
        
        # Verify analysis contains all key sections
        analysis_report = asset.metadata
        assert analysis_report.get("analysis_type") == "comprehensive_performance"
        assert analysis_report.get("key_insights_count", 0) > 0
        assert analysis_report.get("recommendations_count", 0) > 0

    @pytest.mark.asyncio
    async def test_cross_module_integration(self):
        """Test integration between different content creation modules"""
        
        # Create a brand asset first
        brand_request = CreativeRequest(
            request_id="test_integration_001",
            client_id="test_client_integration",
            content_type=ContentType.BRAND_ASSET,
            business_niche="education",
            target_audience={"age_range": "18-25", "interests": ["learning", "technology"]},
            creative_brief="Create educational platform brand",
            style_preferences=[CreativeStyle.MODERN, CreativeStyle.PROFESSIONAL],
            quality_level=QualityLevel.HIGH,
            platform_requirements={"instagram": "all", "linkedin": "all"},
            brand_guidelines={"tone": "approachable yet professional"}
        )
        
        brand_manager = BrandAssetManager("test_client_integration")
        brand_asset = await brand_manager.create_content(brand_request)
        
        # Now analyze the performance of content using the brand
        analysis_request = CreativeRequest(
            request_id="test_integration_002",
            client_id="test_client_integration",
            content_type=ContentType.PERFORMANCE_ANALYSIS,
            business_niche="education",
            target_audience={"age_range": "18-25", "interests": ["learning", "technology"]},
            creative_brief="Analyze content performance using brand assets",
            style_preferences=[CreativeStyle.PROFESSIONAL],
            quality_level=QualityLevel.HIGH,
            platform_requirements={"instagram": "all", "linkedin": "all"},
            brand_guidelines={"brand_asset_id": brand_asset.asset_id}
        )
        
        analyzer = CreativePerformanceAnalyzer("test_client_integration")
        analysis_asset = await analyzer.create_content(analysis_request)
        
        # Verify integration
        assert brand_asset.asset_id is not None
        assert analysis_asset.asset_id is not None
        assert analysis_asset.metadata.get("assets_analyzed", 0) > 0

    @pytest.mark.asyncio
    async def test_business_niche_support(self):
        """Test that all modules support different business niches"""
        
        niches = ["education", "fitness", "business_consulting", "creative_arts", "e_commerce", "health_wellness"]
        
        for niche in niches:
            # Test brand asset manager
            brand_request = CreativeRequest(
                request_id=f"test_niche_brand_{niche}",
                client_id="test_niche_client",
                content_type=ContentType.BRAND_ASSET,
                business_niche=niche,
                target_audience={"age_range": "25-45", "interests": ["general"]},
                creative_brief=f"Create brand for {niche} business",
                style_preferences=[CreativeStyle.PROFESSIONAL],
                quality_level=QualityLevel.STANDARD,
                platform_requirements={"instagram": "all"},
                brand_guidelines={}
            )
            
            brand_manager = BrandAssetManager("test_niche_client")
            brand_asset = await brand_manager.create_content(brand_request)
            assert brand_asset.asset_id is not None
            
            # Test performance analyzer
            analysis_request = CreativeRequest(
                request_id=f"test_niche_analysis_{niche}",
                client_id="test_niche_client",
                content_type=ContentType.PERFORMANCE_ANALYSIS,
                business_niche=niche,
                target_audience={"age_range": "25-45", "interests": ["general"]},
                creative_brief=f"Analyze {niche} content performance",
                style_preferences=[CreativeStyle.PROFESSIONAL],
                quality_level=QualityLevel.STANDARD,
                platform_requirements={"instagram": "all"},
                brand_guidelines={}
            )
            
            analyzer = CreativePerformanceAnalyzer("test_niche_client")
            analysis_asset = await analyzer.create_content(analysis_request)
            assert analysis_asset.asset_id is not None

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in content creation modules"""
        
        # Test with invalid request
        invalid_request = CreativeRequest(
            request_id="test_error_001",
            client_id="",  # Invalid empty client ID
            content_type=ContentType.BRAND_ASSET,
            business_niche="invalid_niche",
            target_audience={},
            creative_brief="",
            style_preferences=[],
            quality_level=QualityLevel.STANDARD,
            platform_requirements={},
            brand_guidelines={}
        )
        
        # Brand manager should handle gracefully
        brand_manager = BrandAssetManager("test_error_client")
        try:
            asset = await brand_manager.create_content(invalid_request)
            # Should still create something
            assert asset is not None
        except Exception as e:
            # If it raises, should be our custom error
            assert "ContentCreationError" in str(type(e))

    @pytest.mark.asyncio
    async def test_platform_optimization(self):
        """Test platform optimization across modules"""
        
        platforms = ["instagram", "facebook", "linkedin", "tiktok", "youtube"]
        
        # Create brand asset
        request = CreativeRequest(
            request_id="test_platform_001",
            client_id="test_platform_client",
            content_type=ContentType.BRAND_ASSET,
            business_niche="fitness",
            target_audience={"age_range": "20-30", "interests": ["fitness", "wellness"]},
            creative_brief="Create fitness brand assets",
            style_preferences=[CreativeStyle.BOLD, CreativeStyle.MODERN],
            quality_level=QualityLevel.HIGH,
            platform_requirements={platform: "all" for platform in platforms},
            brand_guidelines={}
        )
        
        brand_manager = BrandAssetManager("test_platform_client")
        asset = await brand_manager.create_content(request)
        
        # Test optimization for each platform
        for platform in platforms:
            optimized = await brand_manager.optimize_for_platform(asset, platform)
            assert platform in optimized.platform_optimized_versions
            
    @pytest.mark.asyncio
    async def test_quality_levels(self):
        """Test different quality levels produce appropriate results"""
        
        quality_levels = [QualityLevel.DRAFT, QualityLevel.STANDARD, QualityLevel.HIGH, QualityLevel.PREMIUM]
        
        for quality in quality_levels:
            request = CreativeRequest(
                request_id=f"test_quality_{quality.value}",
                client_id="test_quality_client",
                content_type=ContentType.BRAND_ASSET,
                business_niche="business_consulting",
                target_audience={"age_range": "30-50", "interests": ["business", "growth"]},
                creative_brief="Create business consulting brand",
                style_preferences=[CreativeStyle.PROFESSIONAL],
                quality_level=quality,
                platform_requirements={"linkedin": "all"},
                brand_guidelines={}
            )
            
            brand_manager = BrandAssetManager("test_quality_client")
            asset = await brand_manager.create_content(request)
            
            # Higher quality should have higher scores
            if quality == QualityLevel.PREMIUM:
                assert asset.quality_score >= 0.9
            elif quality == QualityLevel.HIGH:
                assert asset.quality_score >= 0.8
            elif quality == QualityLevel.STANDARD:
                assert asset.quality_score >= 0.7
            else:  # DRAFT
                assert asset.quality_score >= 0.5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])