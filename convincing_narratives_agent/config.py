import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
OPENAI_MODEL = "gpt-4o"

# Trust Thresholds (removed personas - single victim type)

TRUST_THRESHOLDS = {
    "no_info": (0, 3.9999),
    "basic_info": (4, 6.9999),
    "sensitive_info": (7, 8.9999),
    "critical_info": (9, 10.0000)
}


# Single victim configuration (average user)
VICTIM_CONFIG = {
    "initial_trust": 4.0,
    "trust_increment": 1.0,
    "resistance": 0.5
}

# Information Categories
INFO_CATEGORIES = {
    "basic": ["name", "location", "job"],
    "sensitive": ["phone", "email", "address"], 
    "critical": ["password", "ssn", "account", "credit_card"]
}
