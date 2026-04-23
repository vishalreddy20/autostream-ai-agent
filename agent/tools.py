# agent/tools.py

import re
from datetime import datetime

def validate_email(email: str) -> bool:
    """
    Validates that the email has a basic valid format.
    Uses a simple regex — not a full RFC 5322 parser.
    This prevents garbage data from being captured as a lead.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates calling a CRM API to store a qualified lead.

    In production (e.g. HubSpot, Salesforce, Zoho CRM), this function
    would make an authenticated HTTP POST request to the CRM endpoint.

    For this assignment, it prints the captured lead and returns a
    structured result dict so the agent can confirm success to the user.

    Args:
        name: Full name of the lead.
        email: Email address (pre-validated).
        platform: Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status, timestamp, and captured data.

    Raises:
        ValueError: If any required field is empty or email is invalid.
    """
    # Guard: all fields must be non-empty
    if not name or not name.strip():
        raise ValueError("Lead name cannot be empty.")
    if not email or not validate_email(email.strip()):
        raise ValueError(f"Invalid email address: {email}")
    if not platform or not platform.strip():
        raise ValueError("Creator platform cannot be empty.")

    # Simulate API call
    timestamp = datetime.now().isoformat()
    print(f"\n{'='*50}")
    print(f"[LEAD CAPTURED] {timestamp}")
    print(f"  Name:     {name.strip()}")
    print(f"  Email:    {email.strip()}")
    print(f"  Platform: {platform.strip()}")
    print(f"{'='*50}\n")

    return {
        "status": "success",
        "timestamp": timestamp,
        "lead": {
            "name": name.strip(),
            "email": email.strip(),
            "platform": platform.strip()
        }
    }
