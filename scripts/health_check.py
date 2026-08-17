#!/usr/bin/env python3
"""Health check script — verify all services are responding."""

import asyncio
import sys

import httpx

SERVICES = {
    "tesseract-ocr": "http://localhost:8001/health",
    "glm-ocr-api": "http://localhost:8002/health",
    "qwen-vl-api": "http://localhost:8003/health",
    "supervisor": "http://localhost:8080/health",
}


async def check_service(name: str, url: str) -> bool:
    """Check if a service is healthy."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                print(f"  ✓ {name}: healthy")
                return True
            else:
                print(f"  ✗ {name}: HTTP {resp.status_code}")
                return False
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


async def main() -> int:
    """Check all services."""
    print("Checking PCB OCR services...")
    results = await asyncio.gather(*[check_service(name, url) for name, url in SERVICES.items()])

    if all(results):
        print("\nAll services healthy.")
        return 0
    else:
        print("\nSome services not ready.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
