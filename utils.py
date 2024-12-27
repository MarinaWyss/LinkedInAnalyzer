import os
import yaml
import time
import logging
from openai import OpenAI, RateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config():
    """Load the configuration file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def openai_request(prompt: str) -> str:
    """Make a request to the OpenAI API with retry logic."""
    config = load_config()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    retries = 0
    delay = config["INITIAL_DELAY"]

    while retries < config["MAX_RETRIES"]:
        try:
            response = client.chat.completions.create(
                model=config["MODEL"],
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            retries += 1
            logger.warning(f"Rate limit reached. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= config["BACKOFF_FACTOR"]

    logger.error("Max retries reached. OpenAI request failed.")
    raise Exception("Max retries reached for OpenAI request.")
