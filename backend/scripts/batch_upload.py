#!/usr/bin/env python3
import os
import requests
from pathlib import Path

API = "http://127.0.0.1:8000/api/detect/upload?camera_id=upload&camera_name=Upload&lat=18.52&lng=73.85"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"

def main():
    files = sorted([p for p in UPLOAD_DIR.iterdir() if p.is_file()])
    if not files:
        print("No files to upload in", UPLOAD_DIR)
        return

    for f in files:
        print(f"Uploading: {f.name}")
        with open(f, "rb") as fh:
            resp = requests.post(API, files={"file": (f.name, fh, "application/octet-stream")})
        print(f"Status: {resp.status_code}")
        try:
            print(resp.json())
        except Exception:
            print(resp.text[:500])
        print("-"*60)

if __name__ == '__main__':
    main()
