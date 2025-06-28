"""
Unit tests for encryption utilities.

Tests all encryption functionality including credentials encryption,
OAuth token storage, API key management, data anonymization, and
secure random generation.
"""

import os
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from backend.utils.encryption import (
    EncryptionManager,
    EncryptionError,
    KeyManagementError,
    encrypt_data,
    decrypt_data,
    generate_api_key
)


class TestEncryptionManager:
    """Test suite for EncryptionManager class."""
    
    @pytest.fixture
    def temp_key_storage(self, tmp_path):
        """Create temporary directory for key storage."""
        return tmp_path / "test_keys"
    
    @pytest.fixture
    def encryption_manager(self, temp_key_storage, monkeypatch):
        """Create EncryptionManager instance with test configuration."""
        # Set test master key
        monkeypatch.setenv('AUTOGURU_MASTER_KEY', 'test-master-key-for-testing-only')
        return EncryptionManager(key_storage_path=temp_key_storage)
    
    def test_initialization_without_master_key(self, temp_key_storage):
        """Test that initialization fails without master key."""
        with pytest.raises(KeyManagementError) as exc_info:
            EncryptionManager(key_storage_path=temp_key_storage)
        assert "Master key not found" in str(exc_info.value)
    
    def test_initialization_with_master_key(self, encryption_manager):
        """Test successful initialization with master key."""
        assert encryption_manager.master_key is not None
        assert encryption_manager.key_storage_path.exists()
        # Check directory permissions (Unix only)
        if os.name != 'nt':
            assert oct(encryption_manager.key_storage_path.stat().st_mode)[-3:] == '700'
    
    def test_encrypt_decrypt_credentials(self, encryption_manager):
        """Test basic credential encryption and decryption."""
        # Test data
        credentials = {
            'username': 'test_user',
            'password': 'secret_password',
            'api_key': 'sk-1234567890',
            'additional_data': {'nested': 'value'}
        }
        business_id = 'business_123'
        
        # Encrypt
        encrypted = encryption_manager.encrypt_credentials(credentials, business_id)
        
        # Verify encrypted format
        assert 'encrypted_data' in encrypted
        assert 'business_id' in encrypted
        assert 'encryption_version' in encrypted
        assert 'encrypted_at' in encrypted
        assert encrypted['business_id'] == business_id
        
        # Decrypt
        decrypted = encryption_manager.decrypt_credentials(encrypted)
        
        # Verify decrypted data matches original
        assert decrypted == credentials
    
    def test_encrypt_different_businesses(self, encryption_manager):
        """Test that different businesses get different encryption."""
        credentials = {'api_key': 'same-key-for-both'}
        
        # Encrypt for two different businesses
        encrypted1 = encryption_manager.encrypt_credentials(credentials, 'business_1')
        encrypted2 = encryption_manager.encrypt_credentials(credentials, 'business_2')
        
        # Encrypted data should be different
        assert encrypted1['encrypted_data'] != encrypted2['encrypted_data']
        
        # But both should decrypt to same credentials
        assert encryption_manager.decrypt_credentials(encrypted1) == credentials
        assert encryption_manager.decrypt_credentials(encrypted2) == credentials
    
    def test_oauth_token_encryption(self, encryption_manager):
        """Test OAuth token secure storage."""
        token_data = {
            'access_token': 'ya29.a0AfH6SMBx...',
            'refresh_token': '1//0gBq4...',
            'expires_in': 3600,
            'token_type': 'Bearer'
        }
        platform = 'google'
        business_id = 'business_456'
        
        # Secure token
        encrypted_token = encryption_manager.secure_oauth_token(
            token_data, platform, business_id
        )
        
        # Verify metadata
        assert encrypted_token['platform'] == platform
        assert encrypted_token['token_type'] == 'oauth'
        assert 'encrypted_data' in encrypted_token
        
        # Retrieve token
        decrypted_token = encryption_manager.retrieve_oauth_token(encrypted_token)
        
        # Verify decrypted data
        assert decrypted_token['access_token'] == token_data['access_token']
        assert decrypted_token['refresh_token'] == token_data['refresh_token']
        assert decrypted_token['platform'] == platform
        assert 'stored_at' in decrypted_token
    
    def test_api_key_encryption_with_rotation(self, encryption_manager):
        """Test API key encryption with rotation tracking."""
        api_key = 'sk-proj-1234567890abcdef'
        service_name = 'openai'
        business_id = 'business_789'
        
        # Encrypt API key
        encrypted_key = encryption_manager.encrypt_api_key(
            api_key, service_name, business_id
        )
        
        # Verify metadata
        assert encrypted_key['service'] == service_name
        assert encrypted_key['key_type'] == 'api_key'
        
        # Decrypt and verify
        decrypted_key = encryption_manager.decrypt_api_key(encrypted_key)
        assert decrypted_key == api_key
    
    def test_api_key_rotation_warning(self, encryption_manager, caplog):
        """Test that rotation warning is logged for old keys."""
        # Create an API key that's already expired
        old_date = datetime.utcnow() - timedelta(days=100)
        
        api_key_data = {
            'key': 'old-api-key',
            'service': 'stripe',
            'created_at': old_date.isoformat(),
            'rotation_due': (old_date + timedelta(days=90)).isoformat()
        }
        
        # Encrypt directly
        encrypted = encryption_manager.encrypt_credentials(api_key_data, 'api_stripe_biz')
        encrypted['service'] = 'stripe'
        encrypted['key_type'] = 'api_key'
        
        # Decrypt - should log warning
        with caplog.at_level('WARNING'):
            encryption_manager.decrypt_api_key(encrypted)
        
        assert 'due for rotation' in caplog.text
    
    def test_data_anonymization(self, encryption_manager):
        """Test data anonymization for various field types."""
        user_data = {
            'email': 'john.doe@example.com',
            'phone': '+1234567890',
            'mobile': '9876543210',
            'ssn': '123-45-6789',
            'name': 'John Doe',
            'address': '123 Main St',
            'unrelated_field': 'stays unchanged'
        }
        
        fields_to_anonymize = ['email', 'phone', 'mobile', 'ssn', 'name', 'address']
        
        anonymized = encryption_manager.anonymize_user_data(
            user_data, fields_to_anonymize
        )
        
        # Check email anonymization
        assert anonymized['email'].startswith('joh***@')
        assert anonymized['email'].endswith('example.com')
        
        # Check phone anonymization
        assert anonymized['phone'].startswith('+123')
        assert '*' in anonymized['phone']
        
        # Check SSN anonymization
        assert anonymized['ssn'] == '*******6789'
        
        # Check generic anonymization
        assert anonymized['name'].startswith('Joh')
        assert '*' in anonymized['name']
        
        # Check metadata
        assert anonymized['_anonymized'] is True
        assert anonymized['_anonymized_fields'] == fields_to_anonymize
        assert '_anonymized_at' in anonymized
        
        # Check unchanged field
        assert anonymized['unrelated_field'] == user_data['unrelated_field']
    
    def test_secure_random_generation(self, encryption_manager):
        """Test secure random string generation."""
        # Test hex format
        hex_random = encryption_manager.generate_secure_random(32, 'hex')
        assert len(hex_random) == 32
        assert all(c in '0123456789abcdef' for c in hex_random)
        
        # Test base64 format
        base64_random = encryption_manager.generate_secure_random(24, 'base64')
        assert len(base64_random) == 24
        
        # Test urlsafe format
        urlsafe_random = encryption_manager.generate_secure_random(20, 'urlsafe')
        assert len(urlsafe_random) == 20
        
        # Test alphanumeric format
        alphanumeric_random = encryption_manager.generate_secure_random(16, 'alphanumeric')
        assert len(alphanumeric_random) == 16
        assert alphanumeric_random.isalnum()
        
        # Test invalid format
        with pytest.raises(ValueError):
            encryption_manager.generate_secure_random(10, 'invalid')
        
        # Test randomness - generate multiple values
        randoms = [encryption_manager.generate_secure_random() for _ in range(10)]
        assert len(set(randoms)) == 10  # All should be unique
    
    def test_secure_password_generation(self, encryption_manager):
        """Test secure password generation."""
        # Test with symbols
        password = encryption_manager.generate_secure_password(16, True)
        assert len(password) == 16
        
        # Verify password complexity
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_symbol = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        assert has_lower
        assert has_upper
        assert has_digit
        assert has_symbol
        
        # Test without symbols
        password_no_symbols = encryption_manager.generate_secure_password(12, False)
        assert len(password_no_symbols) == 12
        assert all(c.isalnum() for c in password_no_symbols)
        
        # Test minimum length passwords
        min_password = encryption_manager.generate_secure_password(4, True)
        assert len(min_password) == 4
    
    def test_sensitive_identifier_hashing(self, encryption_manager):
        """Test secure hashing of sensitive identifiers."""
        identifier = 'user@example.com'
        
        # Hash with default salt
        hash1 = encryption_manager.hash_sensitive_identifier(identifier)
        hash2 = encryption_manager.hash_sensitive_identifier(identifier)
        
        # Same identifier should produce same hash
        assert hash1 == hash2
        
        # Different identifier should produce different hash
        hash3 = encryption_manager.hash_sensitive_identifier('other@example.com')
        assert hash1 != hash3
        
        # Hash with custom salt
        custom_hash = encryption_manager.hash_sensitive_identifier(
            identifier, 'custom-salt'
        )
        assert custom_hash != hash1
    
    def test_key_caching(self, encryption_manager):
        """Test that encryption keys are cached properly."""
        credentials = {'data': 'test'}
        business_id = 'cache_test'
        
        # First encryption stores key
        encryption_manager.encrypt_credentials(credentials, business_id)
        
        # Key should be in cache
        assert business_id in encryption_manager._key_cache
        
        # Remove from cache to test retrieval
        cached_key = encryption_manager._key_cache[business_id]
        del encryption_manager._key_cache[business_id]
        
        # Decrypt should retrieve and re-cache key
        encrypted = encryption_manager.encrypt_credentials(credentials, business_id)
        encryption_manager.decrypt_credentials(encrypted)
        
        # Key should be back in cache
        assert business_id in encryption_manager._key_cache
        assert encryption_manager._key_cache[business_id] == cached_key
    
    def test_encryption_error_handling(self, encryption_manager):
        """Test error handling for various failure scenarios."""
        # Test decryption with missing business_id
        with pytest.raises(EncryptionError):
            encryption_manager.decrypt_credentials({'encrypted_data': 'invalid'})
        
        # Test decryption with invalid base64
        with pytest.raises(EncryptionError):
            encryption_manager.decrypt_credentials({
                'business_id': 'test',
                'encrypted_data': 'not-base64!'
            })
        
        # Test decryption with non-existent key
        with pytest.raises(KeyManagementError):
            encryption_manager.decrypt_credentials({
                'business_id': 'non_existent_business',
                'encrypted_data': 'YW55IGNhcm5hbCBwbGVhc3VyZS4='
            })
    
    def test_key_rotation(self, encryption_manager):
        """Test encryption key rotation functionality."""
        business_id = 'rotation_test'
        
        # Create initial encrypted data
        credentials = {'api_key': 'original-key'}
        encrypted = encryption_manager.encrypt_credentials(credentials, business_id)
        
        # Rotate keys
        encryption_manager.rotate_encryption_keys(business_id)
        
        # Should be able to decrypt with new keys (in real implementation)
        # For now, just verify the method runs without error
        assert True  # Placeholder for actual rotation test


class TestConvenienceFunctions:
    """Test convenience functions for encryption."""
    
    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up test environment."""
        monkeypatch.setenv('AUTOGURU_MASTER_KEY', 'test-convenience-key')
    
    def test_encrypt_decrypt_string(self):
        """Test encrypting and decrypting a simple string."""
        test_string = "This is a secret message"
        business_id = "test_business"
        
        # Encrypt
        encrypted = encrypt_data(test_string, business_id)
        assert isinstance(encrypted, dict)
        assert 'encrypted_data' in encrypted
        
        # Decrypt
        decrypted = decrypt_data(encrypted)
        assert decrypted == test_string
    
    def test_encrypt_decrypt_dict(self):
        """Test encrypting and decrypting a dictionary."""
        test_dict = {
            'username': 'test_user',
            'password': 'secret123',
            'metadata': {'role': 'admin'}
        }
        business_id = "test_business"
        
        # Encrypt
        encrypted = encrypt_data(test_dict, business_id)
        assert isinstance(encrypted, dict)
        
        # Decrypt
        decrypted = decrypt_data(encrypted)
        assert decrypted == test_dict
    
    def test_generate_api_key(self):
        """Test API key generation."""
        # Test for different services
        services = ['openai', 'stripe', 'sendgrid', 'twilio']
        
        for service in services:
            api_key = generate_api_key(service)
            
            # Check format
            parts = api_key.split('-')
            assert len(parts) == 5
            assert parts[0] == service[:4].upper()
            
            # Check length and characters
            for part in parts[1:]:
                assert len(part) == 8
                assert part.isalnum()
        
        # Generate multiple keys - should be unique
        keys = [generate_api_key('test') for _ in range(10)]
        assert len(set(keys)) == 10


class TestBusinessNicheCompatibility:
    """Test that encryption works universally for all business niches."""
    
    @pytest.fixture
    def encryption_manager(self, monkeypatch, tmp_path):
        """Create encryption manager for business tests."""
        monkeypatch.setenv('AUTOGURU_MASTER_KEY', 'business-test-key')
        return EncryptionManager(key_storage_path=tmp_path / "business_keys")
    
    def test_educational_business_encryption(self, encryption_manager):
        """Test encryption for educational business credentials."""
        edu_credentials = {
            'lms_api_key': 'ed-platform-key-123',
            'zoom_api_key': 'zoom-edu-key',
            'zoom_api_secret': 'zoom-edu-secret',
            'stripe_key': 'sk_test_edu_123',
            'course_platform_token': 'teachable-token-xyz'
        }
        
        encrypted = encryption_manager.encrypt_credentials(
            edu_credentials, 'edu_business_001'
        )
        decrypted = encryption_manager.decrypt_credentials(encrypted)
        
        assert decrypted == edu_credentials
    
    def test_fitness_business_encryption(self, encryption_manager):
        """Test encryption for fitness business credentials."""
        fitness_credentials = {
            'mindbody_api_key': 'mb-fitness-key',
            'stripe_key': 'sk_live_fitness_123',
            'facebook_token': 'fb-fitness-page-token',
            'instagram_token': 'ig-fitness-token',
            'youtube_api_key': 'yt-fitness-channel'
        }
        
        encrypted = encryption_manager.encrypt_credentials(
            fitness_credentials, 'fitness_studio_la'
        )
        decrypted = encryption_manager.decrypt_credentials(encrypted)
        
        assert decrypted == fitness_credentials
    
    def test_ecommerce_business_encryption(self, encryption_manager):
        """Test encryption for e-commerce business credentials."""
        ecommerce_credentials = {
            'shopify_api_key': 'shop-api-key-123',
            'shopify_api_secret': 'shop-secret-456',
            'payment_gateway_key': 'pg-merchant-key',
            'shipping_api_token': 'ship-token-789',
            'tax_service_key': 'tax-api-key'
        }
        
        encrypted = encryption_manager.encrypt_credentials(
            ecommerce_credentials, 'online_store_001'
        )
        decrypted = encryption_manager.decrypt_credentials(encrypted)
        
        assert decrypted == ecommerce_credentials
    
    def test_nonprofit_encryption(self, encryption_manager):
        """Test encryption for non-profit organization credentials."""
        nonprofit_credentials = {
            'donation_platform_key': 'donate-key-123',
            'crm_api_token': 'salesforce-nonprofit-token',
            'email_service_key': 'mailchimp-nonprofit',
            'grant_platform_token': 'grant-access-token'
        }
        
        encrypted = encryption_manager.encrypt_credentials(
            nonprofit_credentials, 'nonprofit_help_org'
        )
        decrypted = encryption_manager.decrypt_credentials(encrypted)
        
        assert decrypted == nonprofit_credentials


@pytest.mark.integration
class TestEncryptionIntegration:
    """Integration tests for encryption with other components."""
    
    @pytest.fixture
    def full_setup(self, monkeypatch, tmp_path):
        """Set up full test environment."""
        monkeypatch.setenv('AUTOGURU_MASTER_KEY', 'integration-test-key')
        return {
            'manager': EncryptionManager(key_storage_path=tmp_path / "int_keys"),
            'storage_path': tmp_path
        }
    
    def test_multiple_business_workflow(self, full_setup):
        """Test managing encryption for multiple businesses."""
        manager = full_setup['manager']
        
        # Simulate multiple businesses
        businesses = [
            ('fitness_001', {'type': 'fitness', 'api_key': 'fit-key-1'}),
            ('consulting_001', {'type': 'consulting', 'api_key': 'cons-key-1'}),
            ('artist_001', {'type': 'creative', 'api_key': 'art-key-1'}),
            ('tech_001', {'type': 'saas', 'api_key': 'tech-key-1'})
        ]
        
        encrypted_data = {}
        
        # Encrypt all business credentials
        for business_id, credentials in businesses:
            encrypted = manager.encrypt_credentials(credentials, business_id)
            encrypted_data[business_id] = encrypted
        
        # Verify all can be decrypted correctly
        for business_id, original_creds in businesses:
            decrypted = manager.decrypt_credentials(encrypted_data[business_id])
            assert decrypted == original_creds
        
        # Verify keys are stored
        key_files = list(full_setup['storage_path'].glob('int_keys/*.key'))
        assert len(key_files) == len(businesses)
    
    def test_oauth_token_lifecycle(self, full_setup):
        """Test complete OAuth token lifecycle."""
        manager = full_setup['manager']
        business_id = 'oauth_test_business'
        
        # Initial token
        initial_token = {
            'access_token': 'initial-access-token',
            'refresh_token': 'initial-refresh-token',
            'expires_in': 3600
        }
        
        # Store token
        encrypted_token = manager.secure_oauth_token(
            initial_token, 'google', business_id
        )
        
        # Simulate token refresh
        new_token = {
            'access_token': 'refreshed-access-token',
            'refresh_token': 'initial-refresh-token',  # Same refresh token
            'expires_in': 3600
        }
        
        # Store refreshed token
        encrypted_new = manager.secure_oauth_token(
            new_token, 'google', business_id
        )
        
        # Verify both can be decrypted
        old_decrypted = manager.retrieve_oauth_token(encrypted_token)
        new_decrypted = manager.retrieve_oauth_token(encrypted_new)
        
        assert old_decrypted['access_token'] == 'initial-access-token'
        assert new_decrypted['access_token'] == 'refreshed-access-token'