# SQLAlchemy Metadata Attribute Fix

## Issue Description
The application was failing to start with the following error:
```
sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.
```

This error occurred because SQLAlchemy reserves the `metadata` attribute name for its own internal use when using the Declarative API. Several model classes in the application were using `metadata` as a column name, which conflicted with SQLAlchemy's internal `metadata` attribute.

## Root Cause
The following SQLAlchemy models had `metadata` columns that conflicted with the reserved attribute:
- `PlatformCredentials` (line 79)
- `AdminUser` (line 106) 
- `APIConnectionLog` (line 147)
- `UserPlatformConnection` (line 184)

## Solution Implemented
Renamed all `metadata` columns to `extra_data` to avoid the reserved attribute conflict.

### Files Modified

#### 1. backend/models/admin_models.py
- Changed `metadata = Column(JSON, default=dict)` to `extra_data = Column(JSON, default=dict)` in:
  - `PlatformCredentials` class
  - `AdminUser` class
  - `APIConnectionLog` class
  - `UserPlatformConnection` class
- Updated corresponding `to_dict()` methods to return `extra_data` instead of `metadata`
- Updated Pydantic models:
  - `PlatformCredentialCreate.metadata` → `PlatformCredentialCreate.extra_data`
  - `PlatformCredentialUpdate.metadata` → `PlatformCredentialUpdate.extra_data`
- Updated dataclasses:
  - `PlatformConfiguration.metadata` → `PlatformConfiguration.extra_data`
  - `AIServiceConfiguration.metadata` → `AIServiceConfiguration.extra_data`

#### 2. backend/api/admin_routes.py
- Changed `metadata=credential_data.metadata` to `metadata=credential_data.extra_data` in the `store_platform_credential` function

#### 3. backend/services/credential_manager.py
- Updated SQL INSERT query: `(id, platform_type, credential_name, encrypted_value, created_by, expires_at, metadata)` → `(id, platform_type, credential_name, encrypted_value, created_by, expires_at, extra_data)`
- Updated SQL UPDATE query: `SET encrypted_value = %s, updated_at = %s, expires_at = %s, metadata = %s` → `SET encrypted_value = %s, updated_at = %s, expires_at = %s, extra_data = %s`
- Updated SQL INSERT query for connection logs: `(platform_type, connection_status, response_time_ms, error_message, tested_by, metadata)` → `(platform_type, connection_status, response_time_ms, error_message, tested_by, extra_data)`

## Files NOT Modified
The following files had `metadata` references that were NOT changed because they don't conflict with SQLAlchemy:
- `backend/services/client_service.py` - Uses `metadata` in a dataclass field, not a SQLAlchemy model
- `backend/intelligence/realtime_streaming.py` - Uses `metadata` in a dataclass field, not a SQLAlchemy model
- Content-related files - These access `asset.metadata` on content assets, not SQLAlchemy models

## Database Migration Required
**IMPORTANT**: The database schema needs to be updated to rename the columns from `metadata` to `extra_data`. This will require:

1. Creating a database migration script to rename the columns:
   ```sql
   ALTER TABLE platform_credentials RENAME COLUMN metadata TO extra_data;
   ALTER TABLE admin_users RENAME COLUMN metadata TO extra_data;
   ALTER TABLE api_connection_logs RENAME COLUMN metadata TO extra_data;
   ALTER TABLE user_platform_connections RENAME COLUMN metadata TO extra_data;
   ```

2. Or dropping and recreating the tables if no production data exists.

## Testing
The fix should resolve the SQLAlchemy startup error. The application should now be able to:
- Import the admin models without errors
- Start the FastAPI server successfully
- Create and update platform credentials
- Log API connection tests

## Verification
After deploying this fix, verify:
1. Application starts without SQLAlchemy errors
2. Admin routes work correctly
3. Platform credential storage functions properly
4. Database operations complete successfully