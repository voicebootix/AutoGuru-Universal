#!/usr/bin/env python3
"""
AutoGuru Universal - Encryption Key Generator

This utility generates a valid Fernet encryption key for use in the security configuration.
The generated key ensures all sensitive data is properly encrypted.
"""

import sys
from cryptography.fernet import Fernet


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.
    
    Returns:
        str: A base64-encoded encryption key suitable for Fernet encryption
    """
    return Fernet.generate_key().decode()


def main():
    """Main function to generate and display encryption key."""
    print("AutoGuru Universal - Encryption Key Generator")
    print("=" * 50)
    print()
    
    # Generate the key
    key = generate_encryption_key()
    
    print("Generated Encryption Key:")
    print(key)
    print()
    print("Add this to your .env file as:")
    print(f"SECURITY_ENCRYPTION_KEY={key}")
    print()
    print("⚠️  IMPORTANT: Keep this key secret and secure!")
    print("⚠️  Never commit this key to version control!")
    print("⚠️  Losing this key means losing access to encrypted data!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())