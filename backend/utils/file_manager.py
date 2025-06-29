"""
File Manager Utility

Handles file operations for content assets.
"""

import os
import shutil
import logging
from typing import Dict, List, Any, Optional
import hashlib
import json
from datetime import datetime
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for content assets"""
    
    def __init__(self):
        self.base_assets_path = os.path.join(os.getcwd(), 'assets')
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            'assets',
            'assets/images',
            'assets/videos',
            'assets/audio',
            'assets/documents',
            'assets/temp'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    async def save_file(self, content: bytes, file_path: str) -> str:
        """Save content to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
                
            logger.info(f"Saved file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {str(e)}")
            raise
            
    async def read_file(self, file_path: str) -> bytes:
        """Read file content"""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise
            
    async def copy_file(self, source: str, destination: str) -> str:
        """Copy file to new location"""
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            await asyncio.to_thread(shutil.copy2, source, destination)
            logger.info(f"Copied file from {source} to {destination}")
            return destination
            
        except Exception as e:
            logger.error(f"Failed to copy file: {str(e)}")
            raise
            
    async def move_file(self, source: str, destination: str) -> str:
        """Move file to new location"""
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            await asyncio.to_thread(shutil.move, source, destination)
            logger.info(f"Moved file from {source} to {destination}")
            return destination
            
        except Exception as e:
            logger.error(f"Failed to move file: {str(e)}")
            raise
            
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            raise
            
    def get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        if not os.path.exists(file_path):
            return None
            
        stat = os.stat(file_path)
        return {
            'path': file_path,
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'hash': self.get_file_hash(file_path)
        }
        
    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        temp_dir = os.path.join(self.base_assets_path, 'temp')
        if not os.path.exists(temp_dir):
            return
            
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    await self.delete_file(file_path)
                    
    def get_asset_path(self, asset_type: str, asset_id: str, filename: str) -> str:
        """Generate standardized asset path"""
        return os.path.join(self.base_assets_path, asset_type, asset_id, filename)
        
    async def save_json(self, data: Dict[str, Any], file_path: str):
        """Save JSON data to file"""
        content = json.dumps(data, indent=2).encode('utf-8')
        await self.save_file(content, file_path)
        
    async def read_json(self, file_path: str) -> Dict[str, Any]:
        """Read JSON data from file"""
        content = await self.read_file(file_path)
        return json.loads(content.decode('utf-8'))
        
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern"""
        import glob
        return glob.glob(os.path.join(directory, pattern))
        
    def get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size