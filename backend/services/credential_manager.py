"""
Credential Manager for AutoGuru Universal

This service handles secure storage, encryption, and management of platform
API credentials and configuration settings.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

from backend.models.admin_models import (
    PlatformCredentials,
    SystemConfiguration,
    APIConnectionLog,
    PlatformType,
    AIServiceType,
    CredentialStatus,
    ConnectionStatus,
    PlatformConfiguration,
    AIServiceConfiguration
)
from backend.utils.encryption import EncryptionManager
from backend.database.connection import get_db_context

logger = logging.getLogger(__name__)


class CredentialValidationError(Exception):
    """Raised when credential validation fails"""
    pass


class PlatformConnectionError(Exception):
    """Raised when platform connection fails"""
    pass


class CredentialManager:
    """Manages secure storage and retrieval of platform credentials"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.platform_configs: Dict[PlatformType, PlatformConfiguration] = {}
        self.ai_configs: Dict[AIServiceType, AIServiceConfiguration] = {}
        self._connection_cache: Dict[str, Dict[str, Any]] = {}
        
    async def store_platform_credential(
        self,
        platform_type: PlatformType,
        credential_name: str,
        credential_value: str,
        admin_id: str,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """Store encrypted platform credential"""
        try:
            # Encrypt the credential value
            encrypted_data = self.encryption_manager.encrypt_credentials(
                {'value': credential_value}, f"cred_{platform_type.value}_{credential_name}"
            )
            encrypted_value = encrypted_data['encrypted_data']
            
            async with get_db_context() as session:
                # Check if credential already exists
                existing = await session.execute(
                    "SELECT id FROM platform_credentials WHERE platform_type = %s AND credential_name = %s",
                    (platform_type.value, credential_name)
                )
                
                if existing.fetchone():
                    # Update existing credential
                    credential_id = await self._update_credential(
                        session, platform_type, credential_name, 
                        encrypted_value, admin_id, expires_at, metadata
                    )
                else:
                    # Create new credential
                    credential_id = await self._create_credential(
                        session, platform_type, credential_name,
                        encrypted_value, admin_id, expires_at, metadata
                    )
                
                await session.commit()
                
                # Clear cache for this platform
                cache_key = f"{platform_type.value}_config"
                if cache_key in self._connection_cache:
                    del self._connection_cache[cache_key]
                
                logger.info(f"Stored credential {credential_name} for {platform_type.value}")
                return credential_id
                
        except Exception as e:
            logger.error(f"Failed to store credential: {str(e)}")
            raise CredentialValidationError(f"Failed to store credential: {str(e)}")
    
    async def _create_credential(
        self,
        session,
        platform_type: PlatformType,
        credential_name: str,
        encrypted_value: str,
        admin_id: str,
        expires_at: Optional[datetime],
        metadata: Optional[Dict[str, Any]]
    ) -> UUID:
        """Create new credential record"""
        from uuid import uuid4
        
        credential_id = uuid4()
        
        await session.execute(
            """
            INSERT INTO platform_credentials 
            (id, platform_type, credential_name, encrypted_value, created_by, expires_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                credential_id, platform_type.value, credential_name,
                encrypted_value, admin_id, expires_at,
                json.dumps(metadata or {})
            )
        )
        
        return credential_id
    
    async def _update_credential(
        self,
        session,
        platform_type: PlatformType,
        credential_name: str,
        encrypted_value: str,
        admin_id: str,
        expires_at: Optional[datetime],
        metadata: Optional[Dict[str, Any]]
    ) -> UUID:
        """Update existing credential record"""
        result = await session.execute(
            """
            UPDATE platform_credentials 
            SET encrypted_value = %s, updated_at = %s, expires_at = %s, metadata = %s
            WHERE platform_type = %s AND credential_name = %s
            RETURNING id
            """,
            (
                encrypted_value, datetime.utcnow(), expires_at,
                json.dumps(metadata or {}), platform_type.value, credential_name
            )
        )
        
        row = result.fetchone()
        if not row:
            raise CredentialValidationError("Credential not found for update")
        
        return row[0]
    
    async def get_platform_credential(
        self,
        platform_type: PlatformType,
        credential_name: str
    ) -> Optional[str]:
        """Retrieve and decrypt platform credential"""
        try:
            async with get_db_context() as session:
                result = await session.execute(
                    """
                    SELECT encrypted_value, status, expires_at 
                    FROM platform_credentials 
                    WHERE platform_type = %s AND credential_name = %s
                    """,
                    (platform_type.value, credential_name)
                )
                
                row = result.fetchone()
                if not row:
                    return None
                
                encrypted_value, status, expires_at = row
                
                # Check if credential is expired
                if expires_at and expires_at < datetime.utcnow():
                    logger.warning(f"Credential {credential_name} for {platform_type.value} is expired")
                    return None
                
                # Check if credential is active
                if status != CredentialStatus.ACTIVE.value:
                    logger.warning(f"Credential {credential_name} for {platform_type.value} is {status}")
                    return None
                
                # Decrypt and return
                decrypted_data = self.encryption_manager.decrypt_credentials({
                    'encrypted_data': encrypted_value,
                    'business_id': f"cred_{platform_type.value}_{credential_name}",
                    'encryption_version': '1.0'
                })
                return decrypted_data['value']
                
        except Exception as e:
            logger.error(f"Failed to retrieve credential: {str(e)}")
            return None
    
    async def get_platform_configuration(
        self,
        platform_type: PlatformType,
        force_refresh: bool = False
    ) -> Optional[PlatformConfiguration]:
        """Get complete platform configuration"""
        cache_key = f"{platform_type.value}_config"
        
        # Check cache first
        if not force_refresh and cache_key in self._connection_cache:
            cached_config = self._connection_cache[cache_key]
            if datetime.fromisoformat(cached_config['cached_at']) > datetime.utcnow() - timedelta(minutes=5):
                return PlatformConfiguration(**cached_config['config'])
        
        try:
            # Get all credentials for this platform
            credentials = await self._get_all_platform_credentials(platform_type)
            
            if not credentials:
                return None
            
            # Build configuration object
            config = PlatformConfiguration(platform_type=platform_type)
            
            # Map common credential names
            credential_mapping = {
                'app_id': ['app_id', 'client_id', 'application_id'],
                'app_secret': ['app_secret', 'client_secret', 'application_secret'],
                'access_token': ['access_token', 'bearer_token', 'api_key'],
                'refresh_token': ['refresh_token'],
                'webhook_url': ['webhook_url', 'callback_url']
            }
            
            for config_key, possible_names in credential_mapping.items():
                for name in possible_names:
                    if name in credentials:
                        setattr(config, config_key, credentials[name])
                        break
            
            # Get platform-specific settings
            config.permissions = await self._get_platform_permissions(platform_type)
            config.rate_limits = await self._get_platform_rate_limits(platform_type)
            config.endpoints = await self._get_platform_endpoints(platform_type)
            
            # Cache the configuration
            self._connection_cache[cache_key] = {
                'config': config.__dict__,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to get platform configuration: {str(e)}")
            return None
    
    async def _get_all_platform_credentials(
        self,
        platform_type: PlatformType
    ) -> Dict[str, str]:
        """Get all active credentials for a platform"""
        try:
            async with get_db_context() as session:
                result = await session.execute(
                    """
                    SELECT credential_name, encrypted_value 
                    FROM platform_credentials 
                    WHERE platform_type = %s AND status = %s 
                    AND (expires_at IS NULL OR expires_at > %s)
                    """,
                    (platform_type.value, CredentialStatus.ACTIVE.value, datetime.utcnow())
                )
                
                credentials = {}
                for row in result.fetchall():
                    credential_name, encrypted_value = row
                    try:
                        decrypted_data = self.encryption_manager.decrypt_credentials({
                            'encrypted_data': encrypted_value,
                            'business_id': f"cred_{platform_type.value}_{credential_name}",
                            'encryption_version': '1.0'
                        })
                        credentials[credential_name] = decrypted_data['value']
                    except Exception as e:
                        logger.error(f"Failed to decrypt credential {credential_name}: {str(e)}")
                
                return credentials
                
        except Exception as e:
            logger.error(f"Failed to get platform credentials: {str(e)}")
            return {}
    
    async def test_platform_connection(
        self,
        platform_type: PlatformType,
        admin_id: str,
        test_endpoints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Test platform API connection"""
        start_time = datetime.utcnow()
        
        try:
            # Get platform configuration
            config = await self.get_platform_configuration(platform_type, force_refresh=True)
            if not config:
                raise PlatformConnectionError("Platform configuration not found")
            
            # Perform platform-specific connection test
            test_result = await self._perform_platform_test(config, test_endpoints)
            
            # Calculate response time
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Log the connection attempt
            await self._log_connection_test(
                platform_type, ConnectionStatus.CONNECTED if test_result['success'] else ConnectionStatus.ERROR,
                response_time_ms, test_result.get('error'), admin_id, test_result
            )
            
            return {
                'success': test_result['success'],
                'response_time_ms': response_time_ms,
                'error_message': test_result.get('error'),
                'endpoint_results': test_result.get('endpoint_results', []),
                'tested_at': start_time.isoformat()
            }
            
        except Exception as e:
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            error_message = str(e)
            
            # Log the failed connection attempt
            await self._log_connection_test(
                platform_type, ConnectionStatus.ERROR, response_time_ms, 
                error_message, admin_id, {}
            )
            
            return {
                'success': False,
                'response_time_ms': response_time_ms,
                'error_message': error_message,
                'endpoint_results': [],
                'tested_at': start_time.isoformat()
            }
    
    async def _perform_platform_test(
        self,
        config: PlatformConfiguration,
        test_endpoints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform actual platform API test"""
        import aiohttp
        
        platform_type = config.platform_type
        endpoint_results = []
        
        # Define default test endpoints for each platform
        default_endpoints = {
            PlatformType.FACEBOOK: [
                'https://graph.facebook.com/v18.0/me',
                'https://graph.facebook.com/v18.0/me/accounts'
            ],
            PlatformType.INSTAGRAM: [
                'https://graph.instagram.com/me',
                'https://graph.instagram.com/me/media'
            ],
            PlatformType.TWITTER: [
                'https://api.twitter.com/2/users/me',
                'https://api.twitter.com/2/tweets'
            ],
            PlatformType.LINKEDIN: [
                'https://api.linkedin.com/v2/people/~',
                'https://api.linkedin.com/v2/shares'
            ],
            PlatformType.TIKTOK: [
                'https://open-api.tiktok.com/user/info/',
                'https://open-api.tiktok.com/video/list/'
            ],
            PlatformType.YOUTUBE: [
                'https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true',
                'https://www.googleapis.com/youtube/v3/videos?part=snippet&mine=true'
            ]
        }
        
        endpoints_to_test = test_endpoints or default_endpoints.get(platform_type, [])
        
        if not endpoints_to_test:
            return {'success': False, 'error': 'No test endpoints defined for platform'}
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in endpoints_to_test:
                    try:
                        # Prepare authentication headers
                        headers = await self._get_auth_headers(config, endpoint)
                        
                        # Make request with timeout
                        async with session.get(endpoint, headers=headers, timeout=30) as response:
                            endpoint_result = {
                                'endpoint': endpoint,
                                'status_code': response.status,
                                'success': 200 <= response.status < 300,
                                'response_time_ms': 0,  # Could be measured more precisely
                                'error': None
                            }
                            
                            if not endpoint_result['success']:
                                response_text = await response.text()
                                endpoint_result['error'] = f"HTTP {response.status}: {response_text[:200]}"
                            
                            endpoint_results.append(endpoint_result)
                            
                    except asyncio.TimeoutError:
                        endpoint_results.append({
                            'endpoint': endpoint,
                            'status_code': 0,
                            'success': False,
                            'response_time_ms': 30000,
                            'error': 'Request timeout'
                        })
                    except Exception as e:
                        endpoint_results.append({
                            'endpoint': endpoint,
                            'status_code': 0,
                            'success': False,
                            'response_time_ms': 0,
                            'error': str(e)
                        })
            
            # Determine overall success
            successful_endpoints = sum(1 for result in endpoint_results if result['success'])
            overall_success = successful_endpoints > 0
            
            return {
                'success': overall_success,
                'endpoint_results': endpoint_results,
                'successful_endpoints': successful_endpoints,
                'total_endpoints': len(endpoint_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'endpoint_results': endpoint_results
            }
    
    async def _get_auth_headers(
        self,
        config: PlatformConfiguration,
        endpoint: str
    ) -> Dict[str, str]:
        """Get authentication headers for platform API"""
        headers = {'User-Agent': 'AutoGuru-Universal/1.0'}
        
        if config.platform_type == PlatformType.FACEBOOK:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        elif config.platform_type == PlatformType.INSTAGRAM:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        elif config.platform_type == PlatformType.TWITTER:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        elif config.platform_type == PlatformType.LINKEDIN:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        elif config.platform_type == PlatformType.TIKTOK:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        elif config.platform_type == PlatformType.YOUTUBE:
            if config.access_token:
                headers['Authorization'] = f'Bearer {config.access_token}'
        
        return headers
    
    async def _log_connection_test(
        self,
        platform_type: PlatformType,
        status: ConnectionStatus,
        response_time_ms: int,
        error_message: Optional[str],
        tested_by: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log connection test result"""
        try:
            async with get_db_context() as session:
                await session.execute(
                    """
                    INSERT INTO api_connection_logs 
                    (platform_type, connection_status, response_time_ms, error_message, tested_by, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        platform_type.value, status.value, response_time_ms,
                        error_message, tested_by, json.dumps(metadata)
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to log connection test: {str(e)}")
    
    async def _get_platform_permissions(self, platform_type: PlatformType) -> List[str]:
        """Get required permissions for platform"""
        permissions_map = {
            PlatformType.FACEBOOK: [
                'pages_manage_posts', 'pages_read_engagement', 'pages_show_list'
            ],
            PlatformType.INSTAGRAM: [
                'instagram_basic', 'instagram_content_publish'
            ],
            PlatformType.TWITTER: [
                'tweet.read', 'tweet.write', 'users.read'
            ],
            PlatformType.LINKEDIN: [
                'r_liteprofile', 'r_organization_social', 'w_organization_social'
            ],
            PlatformType.TIKTOK: [
                'user.info.basic', 'video.list', 'video.upload'
            ],
            PlatformType.YOUTUBE: [
                'https://www.googleapis.com/auth/youtube.upload',
                'https://www.googleapis.com/auth/youtube'
            ]
        }
        
        return permissions_map.get(platform_type, [])
    
    async def _get_platform_rate_limits(self, platform_type: PlatformType) -> Dict[str, int]:
        """Get rate limits for platform"""
        rate_limits_map = {
            PlatformType.FACEBOOK: {
                'requests_per_hour': 200,
                'requests_per_day': 5000
            },
            PlatformType.INSTAGRAM: {
                'requests_per_hour': 200,
                'requests_per_day': 5000
            },
            PlatformType.TWITTER: {
                'requests_per_15min': 300,
                'tweets_per_day': 2400
            },
            PlatformType.LINKEDIN: {
                'requests_per_day': 2000,
                'shares_per_day': 150
            },
            PlatformType.TIKTOK: {
                'requests_per_day': 10000,
                'uploads_per_day': 50
            },
            PlatformType.YOUTUBE: {
                'requests_per_day': 10000,
                'uploads_per_day': 6
            }
        }
        
        return rate_limits_map.get(platform_type, {})
    
    async def _get_platform_endpoints(self, platform_type: PlatformType) -> Dict[str, str]:
        """Get API endpoints for platform"""
        endpoints_map = {
            PlatformType.FACEBOOK: {
                'base': 'https://graph.facebook.com/v18.0',
                'pages': 'https://graph.facebook.com/v18.0/me/accounts',
                'post': 'https://graph.facebook.com/v18.0/{page_id}/feed'
            },
            PlatformType.INSTAGRAM: {
                'base': 'https://graph.instagram.com',
                'media': 'https://graph.instagram.com/me/media',
                'post': 'https://graph.instagram.com/me/media'
            },
            PlatformType.TWITTER: {
                'base': 'https://api.twitter.com/2',
                'tweets': 'https://api.twitter.com/2/tweets',
                'users': 'https://api.twitter.com/2/users'
            },
            PlatformType.LINKEDIN: {
                'base': 'https://api.linkedin.com/v2',
                'shares': 'https://api.linkedin.com/v2/shares',
                'organizations': 'https://api.linkedin.com/v2/organizations'
            },
            PlatformType.TIKTOK: {
                'base': 'https://open-api.tiktok.com',
                'upload': 'https://open-api.tiktok.com/video/upload/',
                'user': 'https://open-api.tiktok.com/user/info/'
            },
            PlatformType.YOUTUBE: {
                'base': 'https://www.googleapis.com/youtube/v3',
                'videos': 'https://www.googleapis.com/youtube/v3/videos',
                'channels': 'https://www.googleapis.com/youtube/v3/channels'
            }
        }
        
        return endpoints_map.get(platform_type, {})
    
    async def get_platform_status_overview(self) -> Dict[str, Any]:
        """Get overview of all platform connection statuses"""
        platform_statuses = {}
        
        for platform_type in PlatformType:
            try:
                config = await self.get_platform_configuration(platform_type)
                
                if config:
                    # Check if we have minimum required credentials
                    has_credentials = bool(config.app_id and config.app_secret)
                    
                    # Get recent connection test results
                    recent_test = await self._get_recent_connection_test(platform_type)
                    
                    platform_statuses[platform_type.value] = {
                        'configured': has_credentials,
                        'last_test_status': recent_test.get('status') if recent_test else 'never_tested',
                        'last_test_time': recent_test.get('tested_at') if recent_test else None,
                        'response_time_ms': recent_test.get('response_time_ms') if recent_test else None,
                        'error_message': recent_test.get('error_message') if recent_test else None
                    }
                else:
                    platform_statuses[platform_type.value] = {
                        'configured': False,
                        'last_test_status': 'not_configured',
                        'last_test_time': None,
                        'response_time_ms': None,
                        'error_message': 'Platform not configured'
                    }
                    
            except Exception as e:
                platform_statuses[platform_type.value] = {
                    'configured': False,
                    'last_test_status': 'error',
                    'last_test_time': None,
                    'response_time_ms': None,
                    'error_message': str(e)
                }
        
        return platform_statuses
    
    async def _get_recent_connection_test(
        self,
        platform_type: PlatformType
    ) -> Optional[Dict[str, Any]]:
        """Get most recent connection test for platform"""
        try:
            async with get_db_context() as session:
                result = await session.execute(
                    """
                    SELECT connection_status, response_time_ms, error_message, tested_at
                    FROM api_connection_logs 
                    WHERE platform_type = %s 
                    ORDER BY tested_at DESC 
                    LIMIT 1
                    """,
                    (platform_type.value,)
                )
                
                row = result.fetchone()
                if row:
                    return {
                        'status': row[0],
                        'response_time_ms': row[1],
                        'error_message': row[2],
                        'tested_at': row[3].isoformat() if row[3] else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get recent connection test: {str(e)}")
            return None
    
    async def cleanup_expired_credentials(self) -> int:
        """Clean up expired credentials"""
        try:
            async with get_db_context() as session:
                result = await session.execute(
                    """
                    UPDATE platform_credentials 
                    SET status = %s 
                    WHERE expires_at < %s AND status = %s
                    """,
                    (CredentialStatus.EXPIRED.value, datetime.utcnow(), CredentialStatus.ACTIVE.value)
                )
                
                affected_rows = result.rowcount
                await session.commit()
                
                logger.info(f"Marked {affected_rows} credentials as expired")
                return affected_rows
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired credentials: {str(e)}")
            return 0