from flask import Blueprint, jsonify
import random
import datetime

# --- MARKETING MATH ENGINE ---
# Base Count: 21,912 (as requested)
BASE_COUNT = 21912
# Start Date: Dec 1, 2024
START_DATE = datetime.datetime(2024, 12, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
# Rate: 1.5 forges per hour
RATE_PER_HOUR = 1.5

def calculate_marketing_stats():
    """
    Calculates the 'Vanity Identities Forged' count based on time elapsed.
    Adds a small random jitter to make it look organic.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    elapsed = now - START_DATE
    hours_elapsed = elapsed.total_seconds() / 3600.0

    # Linear Growth
    calculated_count = BASE_COUNT + (hours_elapsed * RATE_PER_HOUR)

    # Dynamic Jitter (Random Integer 0-5)
    jitter = random.randint(0, 5)

    final_count = int(calculated_count + jitter)
    return final_count
