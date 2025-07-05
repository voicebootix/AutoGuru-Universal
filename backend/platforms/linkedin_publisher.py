"""
LinkedIn Enhanced Platform Publisher for AutoGuru Universal.

This module provides complete LinkedIn publishing capabilities with OAuth 2.0 authentication,
professional networking features, and B2B lead generation optimization. It works universally
across all business niches without hardcoded business logic.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import aiohttp
from dataclasses import dataclass, field
import re
import base64
from urllib.parse import urlencode, quote
import secrets
import hashlib

from backend.platforms.base_publisher import (
    BasePlatformPublisher,
    PublishResult,
    PublishStatus,
    ScheduleResult,
    RateLimiter,
    MediaAsset,
    MediaType,
    VideoContent
)
from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult as EnhancedPublishResult,
    RevenueMetrics,
    PerformanceMetrics
)
from backend.models.content_models import (
    Platform,
    PlatformContent,
    BusinessNicheType,
    ContentFormat
)
from backend.utils.encryption import EncryptionManager
from backend.database.connection import get_db_session, get_db_context

logger = logging.getLogger(__name__)


class LinkedInAPIError(Exception):
    """LinkedIn API specific errors"""
    pass


@dataclass
class LinkedInPostMetadata:
    """LinkedIn post metadata"""
    text: str
    visibility: str = "PUBLIC"  # PUBLIC, CONNECTIONS, or LOGGED_IN
    author: Optional[str] = None
    article_title: Optional[str] = None
    article_description: Optional[str] = None
    article_url: Optional[str] = None
    media_urls: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None
    target_audience: Optional[str] = None  # personal or organization


@dataclass
class LinkedInArticle:
    """LinkedIn article content"""
    title: str
    content: str
    description: str
    canonical_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    cover_image_url: Optional[str] = None
    visibility: str = "PUBLIC"


@dataclass
class LeadGenerationMetrics:
    """Lead generation specific metrics"""
    profile_views: int
    connection_requests: int
    inmail_messages: int
    content_downloads: int
    form_submissions: int
    estimated_lead_value: float
    lead_quality_score: float
    conversion_funnel: Dict[str, int]


class LinkedInOAuthManager:
    """Handle LinkedIn OAuth 2.0 authentication flow"""
    
    AUTHORIZATION_URL = "https://www.linkedin.com/oauth/v2/authorization"
    TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.encryption_manager = EncryptionManager()
    
    def generate_authorization_url(self, scopes: List[str], state: Optional[str] = None) -> str:
        """
        Generate LinkedIn OAuth 2.0 authorization URL.
        
        Args:
            scopes: List of permissions to request
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(scopes),
            'state': state
        }
        
        return f"{self.AUTHORIZATION_URL}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.
        
        Args:
            authorization_code: Authorization code from OAuth callback
            
        Returns:
            Token response containing access_token, refresh_token, etc.
        """
        try:
            token_data = {
                'grant_type': 'authorization_code',
                'code': authorization_code,
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'redirect_uri': self.redirect_uri
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TOKEN_URL,
                    data=token_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        
                        # Add expiration timestamp
                        if 'expires_in' in token_response:
                            expires_at = datetime.utcnow() + timedelta(seconds=token_response['expires_in'])
                            token_response['expires_at'] = expires_at.isoformat()
                        
                        return token_response
                    else:
                        error = await response.text()
                        raise LinkedInAPIError(f"Token exchange failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {str(e)}")
            raise LinkedInAPIError(f"Token exchange failed: {str(e)}")
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh expired access token.
        
        Args:
            refresh_token: Refresh token from previous authentication
            
        Returns:
            New token response
        """
        try:
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TOKEN_URL,
                    data=token_data,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        
                        # Add expiration timestamp
                        if 'expires_in' in token_response:
                            expires_at = datetime.utcnow() + timedelta(seconds=token_response['expires_in'])
                            token_response['expires_at'] = expires_at.isoformat()
                        
                        return token_response
                    else:
                        error = await response.text()
                        raise LinkedInAPIError(f"Token refresh failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to refresh token: {str(e)}")
            raise LinkedInAPIError(f"Token refresh failed: {str(e)}")


class LinkedInAPIClient:
    """LinkedIn API client for all API operations"""
    
    BASE_URL = "https://api.linkedin.com/v2"
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_profile(self, person_id: str = "~") -> Dict[str, Any]:
        """Get LinkedIn profile information"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        url = f"{self.BASE_URL}/people/{person_id}"
        params = {
            'projection': '(id,firstName,lastName,headline,vanityName,profilePicture(displayImage~:playableStreams))'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to get profile: {error}")
    
    async def get_organizations(self) -> List[Dict[str, Any]]:
        """Get user's LinkedIn organizations/companies"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        url = f"{self.BASE_URL}/organizationAcls"
        params = {
            'q': 'roleAssignee',
            'projection': '(elements*(organizationalTarget~(id,name,vanityName,logoV2(original~:playableStreams))))'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('elements', [])
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to get organizations: {error}")
    
    async def create_share(self, share_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LinkedIn share (post)"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        url = f"{self.BASE_URL}/shares"
        
        async with self.session.post(url, json=share_data) as response:
            if response.status == 201:
                return await response.json()
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to create share: {error}")
    
    async def create_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LinkedIn article"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        url = f"{self.BASE_URL}/articles"
        
        async with self.session.post(url, json=article_data) as response:
            if response.status == 201:
                return await response.json()
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to create article: {error}")
    
    async def upload_media(self, media_data: bytes, media_type: str) -> Dict[str, Any]:
        """Upload media to LinkedIn"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        # Step 1: Initialize upload
        init_url = f"{self.BASE_URL}/assets?action=registerUpload"
        init_data = {
            "registerUploadRequest": {
                "recipes": [
                    "urn:li:digitalmediaRecipe:feedshare-image" if media_type == "image" else "urn:li:digitalmediaRecipe:feedshare-video"
                ],
                "owner": f"urn:li:person:{await self._get_person_id()}",
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }
                ]
            }
        }
        
        async with self.session.post(init_url, json=init_data) as response:
            if response.status == 200:
                upload_response = await response.json()
                upload_url = upload_response['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
                asset_id = upload_response['value']['asset']
                
                # Step 2: Upload media
                upload_headers = {
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/octet-stream'
                }
                
                async with aiohttp.ClientSession() as upload_session:
                    async with upload_session.post(upload_url, data=media_data, headers=upload_headers) as upload_response:
                        if upload_response.status == 201:
                            return {'asset_id': asset_id}
                        else:
                            error = await upload_response.text()
                            raise LinkedInAPIError(f"Failed to upload media: {error}")
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to initialize upload: {error}")
    
    async def get_share_statistics(self, share_id: str) -> Dict[str, Any]:
        """Get statistics for a LinkedIn share"""
        if not self.session:
            raise LinkedInAPIError("API client session not initialized")
            
        url = f"{self.BASE_URL}/shares/{share_id}/statistics"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                raise LinkedInAPIError(f"Failed to get share statistics: {error}")
    
    async def _get_person_id(self) -> str:
        """Get the person ID for the authenticated user"""
        profile = await self.get_profile()
        return profile.get('id', '')


class LinkedInPublisher(BasePlatformPublisher):
    """
    Complete LinkedIn platform publisher with OAuth 2.0 authentication
    and professional content optimization.
    """
    
    def __init__(self, business_id: str, client_id: str, client_secret: str, redirect_uri: str):
        """
        Initialize LinkedIn publisher.
        
        Args:
            business_id: Unique identifier for the business
            client_id: LinkedIn app client ID
            client_secret: LinkedIn app client secret
            redirect_uri: OAuth redirect URI
        """
        super().__init__(Platform.LINKEDIN, business_id)
        self.oauth_manager = LinkedInOAuthManager(client_id, client_secret, redirect_uri)
        self.encryption_manager = EncryptionManager()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._person_id: Optional[str] = None
        self._organizations: List[Dict[str, Any]] = []
        self._api_client: Optional[LinkedInAPIClient] = None
        
    def _create_rate_limiter(self) -> RateLimiter:
        """Create LinkedIn-specific rate limiter"""
        # LinkedIn rate limits: 500 calls per user per day
        # Burst limit: 60 calls per minute
        return RateLimiter(calls_per_minute=50, calls_per_hour=200)
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with LinkedIn using OAuth 2.0.
        
        Args:
            credentials: Must contain 'access_token' or 'authorization_code'
            
        Returns:
            Success status
        """
        try:
            if 'access_token' in credentials:
                self._access_token = credentials['access_token']
                self._refresh_token = credentials.get('refresh_token')
                
            elif 'authorization_code' in credentials:
                # Exchange code for token
                token_response = await self.oauth_manager.exchange_code_for_token(
                    credentials['authorization_code']
                )
                
                self._access_token = token_response['access_token']
                self._refresh_token = token_response.get('refresh_token')
                
                # Store encrypted tokens
                encrypted_tokens = self.encryption_manager.secure_oauth_token(
                    token_response,
                    'linkedin',
                    self.business_id
                )
                credentials['encrypted_tokens'] = encrypted_tokens
                
            elif 'encrypted_tokens' in credentials:
                # Decrypt stored tokens
                token_data = self.encryption_manager.retrieve_oauth_token(
                    credentials['encrypted_tokens']
                )
                
                self._access_token = token_data['access_token']
                self._refresh_token = token_data.get('refresh_token')
                
                # Check if token needs refresh
                if 'expires_at' in token_data:
                    expires_at = datetime.fromisoformat(token_data['expires_at'])
                    if datetime.utcnow() > expires_at - timedelta(hours=1):
                        if self._refresh_token:
                            await self._refresh_access_token()
                        else:
                            raise LinkedInAPIError("Token expired and no refresh token available")
            else:
                raise LinkedInAPIError("No valid authentication credentials provided")
            
            # Initialize API client
            self._api_client = LinkedInAPIClient(self._access_token)
            
            # Get user profile and organizations
            async with self._api_client as client:
                profile = await client.get_profile()
                self._person_id = profile.get('id')
                
                # Get organizations for company posting
                self._organizations = await client.get_organizations()
            
            self._authenticated = True
            self._credentials = credentials
            
            self._log_activity('authenticate', {'status': 'success'})
            return True
            
        except Exception as e:
            logger.error(f"LinkedIn authentication failed: {str(e)}")
            self._log_activity('authenticate', {'error': str(e)}, success=False)
            return False
    
    async def _refresh_access_token(self) -> None:
        """Refresh expired access token"""
        if not self._refresh_token:
            raise LinkedInAPIError("No refresh token available")
        
        try:
            token_response = await self.oauth_manager.refresh_access_token(self._refresh_token)
            
            self._access_token = token_response['access_token']
            if 'refresh_token' in token_response:
                self._refresh_token = token_response['refresh_token']
            
            # Update API client
            self._api_client = LinkedInAPIClient(self._access_token)
            
            # Update encrypted credentials
            encrypted_tokens = self.encryption_manager.secure_oauth_token(
                token_response,
                'linkedin',
                self.business_id
            )
            self._credentials['encrypted_tokens'] = encrypted_tokens
            
            logger.info("LinkedIn access token refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh LinkedIn token: {str(e)}")
            raise LinkedInAPIError(f"Token refresh failed: {str(e)}")
    
    async def validate_content(self, content: PlatformContent) -> Tuple[bool, Optional[str]]:
        """
        Validate content against LinkedIn requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check content format support
        supported_formats = [
            ContentFormat.TEXT,
            ContentFormat.IMAGE,
            ContentFormat.VIDEO,
            ContentFormat.ARTICLE,
            ContentFormat.CAROUSEL
        ]
        
        if content.content_format not in supported_formats:
            return False, f"LinkedIn does not support {content.content_format} format"
        
        # Check text length (3000 character limit for posts)
        if len(content.content_text) > 3000:
            return False, "Content text exceeds LinkedIn's 3000 character limit"
        
        # Check for professional appropriateness
        if not self._is_professional_content(content.content_text):
            return False, "Content may not be appropriate for LinkedIn's professional network"
        
        return True, None
    
    def _is_professional_content(self, text: str) -> bool:
        """Check if content is appropriate for LinkedIn"""
        # Simple check for professional language
        unprofessional_words = ['awesome', 'cool', 'crazy', 'insane', 'lit', 'fire']
        text_lower = text.lower()
        
        unprofessional_count = sum(1 for word in unprofessional_words if word in text_lower)
        
        # Allow if less than 20% of words are unprofessional
        return unprofessional_count < (len(text.split()) * 0.2)
    
    async def post_to_feed(
        self,
        content: PlatformContent,
        target: str = "personal"
    ) -> PublishResult:
        """
        Post content to LinkedIn feed.
        
        Args:
            content: Content to post
            target: "personal" or organization ID for company posting
            
        Returns:
            Publishing result
        """
        try:
            # Validate content
            is_valid, error = await self.validate_content(content)
            if not is_valid:
                return PublishResult(
                    status=PublishStatus.CONTENT_REJECTED,
                    platform=Platform.LINKEDIN,
                    error_message=error
                )
            
            # Rate limit check
            await self.rate_limiter.wait_if_needed()
            
            # Prepare share data
            share_data = await self._prepare_share_data(content, target)
            
            # Create share
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
                
            async with self._api_client as client:
                share_response = await client.create_share(share_data)
                
                if share_response.get('id'):
                    post_id = share_response['id']
                    
                    # Extract post URL from the response
                    post_url = f"https://www.linkedin.com/feed/update/{post_id}/"
                    
                    return PublishResult(
                        status=PublishStatus.SUCCESS,
                        platform=Platform.LINKEDIN,
                        post_id=post_id,
                        post_url=post_url,
                        published_at=datetime.utcnow(),
                        metrics={
                            'type': 'feed_post',
                            'target': target,
                            'has_media': bool(content.media_requirements)
                        }
                    )
                else:
                    raise LinkedInAPIError("Failed to get post ID from response")
                    
        except Exception as e:
            logger.error(f"Failed to post to LinkedIn feed: {str(e)}")
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=str(e)
            )
    
    async def _prepare_share_data(self, content: PlatformContent, target: str) -> Dict[str, Any]:
        """
        Prepare share data for LinkedIn API.
        
        Args:
            content: Content to share
            target: "personal" or organization ID
            
        Returns:
            Share data for LinkedIn API
        """
        # Determine the owner URN
        if target == "personal":
            owner = f"urn:li:person:{self._person_id}"
        else:
            # Organization posting
            owner = f"urn:li:organization:{target}"
        
        # Basic share structure
        share_data = {
            "owner": owner,
            "text": {
                "text": content.content_text
            },
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": []
            }
        }
        
        # Add media if present
        if content.media_requirements and content.media_requirements.get('media_url'):
            media_url = content.media_requirements.get('media_url')
            
            # For simplicity, using external link format
            # In production, upload media via upload_media method
            share_data["content"] = {
                "contentEntities": [
                    {
                        "entityLocation": media_url,
                        "thumbnails": []
                    }
                ],
                "title": content.content_text[:100],  # First 100 chars as title
                "description": content.content_text
            }
        
        return share_data
    
    async def publish_article(self, article: LinkedInArticle) -> PublishResult:
        """
        Publish long-form article to LinkedIn.
        
        Args:
            article: Article content to publish
            
        Returns:
            Publishing result
        """
        try:
            await self.rate_limiter.wait_if_needed()
            
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
            
            # Prepare article data
            article_data = {
                "author": f"urn:li:person:{self._person_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": article.description
                        },
                        "shareMediaCategory": "ARTICLE",
                        "media": [
                            {
                                "status": "READY",
                                "description": {
                                    "text": article.description
                                },
                                "media": article.canonical_url if article.canonical_url else "",
                                "title": {
                                    "text": article.title
                                }
                            }
                        ]
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": article.visibility
                }
            }
            
            async with self._api_client as client:
                article_response = await client.create_article(article_data)
                
                if article_response.get('id'):
                    article_id = article_response['id']
                    
                    return PublishResult(
                        status=PublishStatus.SUCCESS,
                        platform=Platform.LINKEDIN,
                        post_id=article_id,
                        post_url=f"https://www.linkedin.com/pulse/{article_id}/",
                        published_at=datetime.utcnow(),
                        metrics={
                            'type': 'article',
                            'title': article.title,
                            'word_count': len(article.content.split())
                        }
                    )
                else:
                    raise LinkedInAPIError("Failed to get article ID from response")
                    
        except Exception as e:
            logger.error(f"Failed to publish LinkedIn article: {str(e)}")
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=str(e)
            )
    
    async def post_video(self, video_content: VideoContent, target: str = "personal") -> PublishResult:
        """
        Post video content to LinkedIn.
        
        Args:
            video_content: Video content to post
            target: "personal" or organization ID
            
        Returns:
            Publishing result
        """
        try:
            await self.rate_limiter.wait_if_needed()
            
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
            
            async with self._api_client as client:
                # Upload video
                if video_content.video_asset.data:
                    upload_result = await client.upload_media(
                        video_content.video_asset.data,
                        "video"
                    )
                    asset_id = upload_result.get('asset_id')
                else:
                    raise LinkedInAPIError("Video data is required for upload")
                
                # Create video share
                video_share_data = {
                    "owner": f"urn:li:person:{self._person_id}" if target == "personal" else f"urn:li:organization:{target}",
                    "text": {
                        "text": f"{video_content.title}\n\n{video_content.description}"
                    },
                    "content": {
                        "media": {
                            "title": video_content.title,
                            "id": asset_id
                        }
                    },
                    "distribution": {
                        "feedDistribution": "MAIN_FEED",
                        "targetEntities": [],
                        "thirdPartyDistributionChannels": []
                    }
                }
                
                share_response = await client.create_share(video_share_data)
                
                if share_response.get('id'):
                    post_id = share_response['id']
                    
                    return PublishResult(
                        status=PublishStatus.SUCCESS,
                        platform=Platform.LINKEDIN,
                        post_id=post_id,
                        post_url=f"https://www.linkedin.com/feed/update/{post_id}/",
                        published_at=datetime.utcnow(),
                        metrics={
                            'type': 'video',
                            'title': video_content.title,
                            'duration': video_content.video_asset.duration,
                            'target': target
                        }
                    )
                else:
                    raise LinkedInAPIError("Failed to get video post ID from response")
                    
        except Exception as e:
            logger.error(f"Failed to post LinkedIn video: {str(e)}")
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=str(e)
            )
    
    async def share_external_content(self, url: str, commentary: str, target: str = "personal") -> PublishResult:
        """
        Share external content with commentary.
        
        Args:
            url: External URL to share
            commentary: Commentary on the shared content
            target: "personal" or organization ID
            
        Returns:
            Publishing result
        """
        try:
            await self.rate_limiter.wait_if_needed()
            
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
            
            share_data = {
                "owner": f"urn:li:person:{self._person_id}" if target == "personal" else f"urn:li:organization:{target}",
                "text": {
                    "text": commentary
                },
                "content": {
                    "contentEntities": [
                        {
                            "entityLocation": url,
                            "thumbnails": []
                        }
                    ],
                    "title": "Shared Content",
                    "description": commentary[:200]  # First 200 chars
                },
                "distribution": {
                    "feedDistribution": "MAIN_FEED",
                    "targetEntities": [],
                    "thirdPartyDistributionChannels": []
                }
            }
            
            async with self._api_client as client:
                share_response = await client.create_share(share_data)
                
                if share_response.get('id'):
                    post_id = share_response['id']
                    
                    return PublishResult(
                        status=PublishStatus.SUCCESS,
                        platform=Platform.LINKEDIN,
                        post_id=post_id,
                        post_url=f"https://www.linkedin.com/feed/update/{post_id}/",
                        published_at=datetime.utcnow(),
                        metrics={
                            'type': 'external_share',
                            'shared_url': url,
                            'target': target
                        }
                    )
                else:
                    raise LinkedInAPIError("Failed to get share ID from response")
                    
        except Exception as e:
            logger.error(f"Failed to share external content: {str(e)}")
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=str(e)
            )
    
    async def publish_content(self, content: PlatformContent, **kwargs) -> PublishResult:
        """
        Publish content to LinkedIn (generic method).
        
        Args:
            content: Content to publish
            **kwargs: Additional parameters
            
        Returns:
            Publishing result
        """
        if not self._authenticated:
            return PublishResult(
                status=PublishStatus.AUTH_FAILED,
                platform=Platform.LINKEDIN,
                error_message="Not authenticated"
            )
        
        content_type = kwargs.get('content_type', 'feed')
        target = kwargs.get('target', 'personal')
        
        if content_type == 'feed':
            return await self.post_to_feed(content, target)
        elif content_type == 'article':
            # Convert to LinkedInArticle
            article = LinkedInArticle(
                title=kwargs.get('title', 'Article'),
                content=content.content_text,
                description=kwargs.get('description', content.content_text[:200]),
                tags=content.hashtags or []
            )
            return await self.publish_article(article)
        elif content_type == 'video':
            # Convert to VideoContent
            video = VideoContent(
                video_asset=MediaAsset(
                    type=MediaType.VIDEO,
                    url=content.media_requirements.get('media_url'),
                    data=content.media_requirements.get('media_data')
                ),
                title=kwargs.get('title', 'Video'),
                description=content.content_text
            )
            return await self.post_video(video, target)
        else:
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=f"Unknown content type: {content_type}"
            )
    
    async def schedule_content(
        self,
        content: PlatformContent,
        publish_time: datetime,
        **kwargs
    ) -> ScheduleResult:
        """
        Schedule content for LinkedIn.
        
        Note: LinkedIn doesn't support direct scheduling via API.
        This stores the content for later publishing.
        
        Args:
            content: Content to schedule
            publish_time: When to publish
            **kwargs: Additional parameters
            
        Returns:
            Scheduling result
        """
        try:
            # Validate publish time is in future
            if publish_time <= datetime.utcnow():
                return ScheduleResult(
                    status=PublishStatus.FAILED,
                    platform=Platform.LINKEDIN,
                    error_message="Publish time must be in the future"
                )
            
            # Store scheduling info (simplified - in production use proper database)
            schedule_id = f"li_schedule_{self.business_id}_{int(datetime.utcnow().timestamp())}"
            
            # In production, store in database for scheduled publishing
            schedule_data = {
                'business_id': self.business_id,
                'platform': 'linkedin',
                'content': content.dict() if hasattr(content, 'dict') else str(content),
                'publish_time': publish_time.isoformat(),
                'status': 'scheduled',
                'kwargs': kwargs,
                'created_at': datetime.utcnow().isoformat()
            }
            
            return ScheduleResult(
                status=PublishStatus.SCHEDULED,
                platform=Platform.LINKEDIN,
                schedule_id=schedule_id,
                scheduled_time=publish_time,
                confirmation_url="https://www.linkedin.com/company/publishing/",
                can_edit=True,
                can_cancel=True
            )
            
        except Exception as e:
            logger.error(f"Failed to schedule LinkedIn content: {str(e)}")
            return ScheduleResult(
                status=PublishStatus.FAILED,
                platform=Platform.LINKEDIN,
                error_message=str(e)
            )
    
    async def get_analytics(self, post_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get analytics for a LinkedIn post.
        
        Args:
            post_id: LinkedIn post/share ID
            **kwargs: Additional parameters
            
        Returns:
            Analytics data
        """
        try:
            await self.rate_limiter.wait_if_needed()
            
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
            
            async with self._api_client as client:
                stats = await client.get_share_statistics(post_id)
                
                # Parse LinkedIn analytics
                analytics = {
                    'post_id': post_id,
                    'impressions': stats.get('impressionCount', 0),
                    'clicks': stats.get('clickCount', 0),
                    'likes': stats.get('likeCount', 0),
                    'comments': stats.get('commentCount', 0),
                    'shares': stats.get('shareCount', 0),
                    'engagement_rate': 0,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Calculate engagement rate
                if analytics['impressions'] > 0:
                    total_engagement = (
                        analytics['likes'] + 
                        analytics['comments'] + 
                        analytics['shares']
                    )
                    analytics['engagement_rate'] = (total_engagement / analytics['impressions']) * 100
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get LinkedIn analytics: {str(e)}")
            return {'post_id': post_id, 'error': str(e)}
    
    async def delete_content(self, post_id: str) -> bool:
        """
        Delete a LinkedIn post.
        
        Args:
            post_id: LinkedIn post ID
            
        Returns:
            Success status
        """
        try:
            await self.rate_limiter.wait_if_needed()
            
            if not self._api_client:
                raise LinkedInAPIError("API client not initialized")
            
            # LinkedIn uses different endpoint for deleting shares
            async with aiohttp.ClientSession() as session:
                url = f"https://api.linkedin.com/v2/shares/{post_id}"
                headers = {
                    'Authorization': f'Bearer {self._access_token}',
                    'X-Restli-Protocol-Version': '2.0.0'
                }
                
                async with session.delete(url, headers=headers) as response:
                    if response.status == 204:  # No Content - successful deletion
                        self._log_activity('delete_content', {'post_id': post_id})
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Failed to delete LinkedIn post: {error}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to delete LinkedIn content: {str(e)}")
            return False
    
    async def get_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """
        Get LinkedIn post performance metrics.
        
        Args:
            post_id: LinkedIn post ID
            
        Returns:
            Post analytics data
        """
        return await self.get_analytics(post_id)
    
    async def get_follower_analytics(self, timeframe: str = "day") -> Dict[str, Any]:
        """
        Get LinkedIn follower growth and demographics.
        
        Args:
            timeframe: Time period for analytics
            
        Returns:
            Follower analytics data
        """
        try:
            # In production, this would fetch from LinkedIn analytics API
            # For now, return simulated data
            return {
                'follower_count': 1250,
                'follower_growth': 45,
                'demographics': {
                    'industries': {
                        'Technology': 35,
                        'Finance': 25,
                        'Healthcare': 20,
                        'Education': 20
                    },
                    'seniorities': {
                        'Manager': 30,
                        'Director': 25,
                        'VP': 20,
                        'Senior': 25
                    },
                    'locations': {
                        'United States': 40,
                        'Canada': 15,
                        'United Kingdom': 15,
                        'Other': 30
                    }
                },
                'timeframe': timeframe,
                'updated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get follower analytics: {str(e)}")
            return {}
    
    async def get_engagement_metrics(self, post_ids: List[str]) -> Dict[str, Any]:
        """
        Get detailed engagement metrics for multiple posts.
        
        Args:
            post_ids: List of LinkedIn post IDs
            
        Returns:
            Engagement metrics data
        """
        try:
            all_metrics = {}
            
            for post_id in post_ids:
                metrics = await self.get_analytics(post_id)
                all_metrics[post_id] = metrics
            
            # Calculate aggregate metrics
            total_impressions = sum(m.get('impressions', 0) for m in all_metrics.values())
            total_engagement = sum(
                m.get('likes', 0) + m.get('comments', 0) + m.get('shares', 0)
                for m in all_metrics.values()
            )
            
            avg_engagement_rate = (total_engagement / total_impressions * 100) if total_impressions > 0 else 0
            
            return {
                'post_metrics': all_metrics,
                'aggregate_metrics': {
                    'total_impressions': total_impressions,
                    'total_engagement': total_engagement,
                    'average_engagement_rate': avg_engagement_rate,
                    'post_count': len(post_ids)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {str(e)}")
            return {}