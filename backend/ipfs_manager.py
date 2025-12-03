import hashlib
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class IPFSManager:
    def __init__(self):
        self.api_key = os.getenv('PINATA_API_KEY')
        self.secret_key = os.getenv('PINATA_SECRET_API_KEY')
        
        if not self.api_key or not self.secret_key:
            print("⚠️ Warning: PINATA_API_KEY or PINATA_SECRET_API_KEY not found in environment variables.")
    
    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA-256 hash of a local file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def upload_to_ipfs(self, file_path: str, metadata: dict = None) -> dict:
        """Upload file to IPFS using Pinata REST API directly"""
        if not self.api_key or not self.secret_key:
            return {"error": "Pinata credentials not configured"}

        # 1. Compute Integrity Hash
        sha256_hash = self.compute_sha256(file_path)
        
        # 2. Prepare Metadata (convert all values to strings)
        keyvalues = {
            "sha256": sha256_hash,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            for key, value in metadata.items():
                keyvalues[key] = str(value)
        
        pin_metadata = {
            "name": os.path.basename(file_path),
            "keyvalues": keyvalues
        }
        
        # 3. Upload to Pinata using REST API directly
        try:
            url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
            
            headers = {
                "pinata_api_key": self.api_key,
                "pinata_secret_api_key": self.secret_key
            }
            
            # Open file and prepare multipart form data
            with open(file_path, 'rb') as file:
                files = {
                    'file': (os.path.basename(file_path), file, 'image/jpeg')
                }
                
                data = {
                    'pinataMetadata': json.dumps(pin_metadata)
                }
                
                response = requests.post(url, files=files, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result.get('IpfsHash')
                
                if not ipfs_hash:
                    return {"error": f"No IpfsHash in response: {result}"}
                
                return {
                    "success": True,
                    "ipfs_hash": ipfs_hash,
                    "sha256_hash": sha256_hash,
                    "url": f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}",
                    "timestamp": result.get('Timestamp'),
                    "size": result.get('PinSize')
                }
            else:
                return {"error": f"Upload failed with status {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}

    def verify_integrity(self, ipfs_hash: str, stored_sha256: str) -> dict:
        """Download from IPFS and verify SHA-256 hash matches"""
        url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                return {"verified": False, "error": f"Download failed (Status {response.status_code})"}
                
            # Compute hash of downloaded content
            current_hash = hashlib.sha256(response.content).hexdigest()
            
            is_valid = (current_hash == stored_sha256)
            return {
                "verified": is_valid,
                "stored_hash": stored_sha256,
                "current_hash": current_hash,
                "match": "✅ MATCH" if is_valid else "❌ MISMATCH"
            }
        except Exception as e:
            return {"verified": False, "error": str(e)}
