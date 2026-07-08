    

import hashlib


def compute_file_hash(file_path):
    # Initialize the hashing algorithm
    sha256_hash = hashlib.sha256()
    
    # Read and update the hash object in 64KB chunks
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()