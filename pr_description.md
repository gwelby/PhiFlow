🔒 [Fix Missing Authentication on Sensitive Endpoints]

🎯 **What:**
The `/process_audio` and `/process_video` endpoints in `src/_archive/quantum_web.py` previously accepted data without verifying the caller's identity. This PR adds API key authentication to both endpoints using FastAPI's dependency injection (`Security(get_api_key)`).

⚠️ **Risk:**
Without authentication, these endpoints were open to the public. An attacker could send arbitrary or oversized payloads to trigger complex quantum field simulations, causing high CPU/GPU load, potentially leading to a denial-of-service (DoS) condition or unauthorized usage of the service.

🛡️ **Solution:**
- Configured a new FastAPI security dependency `get_api_key` that parses the `X-API-Key` HTTP header.
- The API key is validated against the `QUANTUM_API_KEY` environment variable.
- Used `secrets.compare_digest` for the string comparison to prevent timing attacks.
- If the `QUANTUM_API_KEY` is not set on the server, the server will log a warning and return a `500 Internal Server Error` to fail securely without exposing functionality.
- If the provided key is missing or incorrect, the server returns a `401 Unauthorized` status.
