import os
from ipfs_manager import IPFSManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ipfs():
    try:
        print("Initializing IPFS Manager...")
        ipfs = IPFSManager()
        
        # Create a dummy file
        test_file = "test_evidence.txt"
        with open(test_file, "w") as f:
            f.write("This is a test evidence file for SecureView IPFS integration.")
            
        print(f"Uploading {test_file} to Pinata...")
        result = ipfs.upload_to_ipfs(test_file, metadata={"type": "test_verification"})
        
        if result:
            print("\n✅ Upload Successful!")
            print(f"IPFS Hash: {result['ipfs_hash']}")
            print(f"SHA256: {result['sha256_hash']}")
            print(f"URL: {result['url']}")
        else:
            print("\n❌ Upload Failed (No result returned)")
            
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_ipfs()
