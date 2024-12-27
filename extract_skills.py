import os
import csv
import logging
import requests
from dotenv import load_dotenv
from typing import List, Dict
from utils import openai_request, load_config

# Load configuration
config = load_config()
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Files
QUERIED_IDS_FILE = "data/queried_job_ids.csv"
OUTPUT_FILE = "data/job_data.csv"

# LinkedIn API Parameters
LINKEDIN_API_KEY = config['LINKEDIN_API_KEY']
LINKEDIN_BASE_URL = "https://api.scrapingdog.com/linkedinjobs"

# Location IDs to iterate
LOCATION_IDS = [
    "101318387", "106315325", "101098412", "101949407", "104383890", "101174742", "90009540",
    "103366113", "100025096", "102199904", "101282230", "103035651", "105015875", "90009659",
    "101165590", "90009496", "104738515", "90009824", "100752109", "91000000", "91000002", "91000007",
    "90010383", "90010409", "101452733", "100992797"
]

def load_queried_ids(file_path: str) -> set:
    """Load previously queried job IDs."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, "r") as file:
        return set(line.strip() for line in file)

def save_queried_ids(file_path: str, job_ids: List[str]):
    """Save queried job IDs to file."""
    with open(file_path, "a") as file:
        writer = csv.writer(file)
        for job_id in job_ids:
            writer.writerow([job_id])

def fetch_job_listings(field: str, location_id: str, page: int = 1) -> List[Dict]:
    """Fetch job listings from LinkedIn API."""
    params = {
        "api_key": LINKEDIN_API_KEY,
        "field": field,
        "geoid": location_id,
        "page": page,
    }
    response = requests.get(LINKEDIN_BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def fetch_job_details(job_id: str) -> Dict:
    """Fetch job details for a specific job ID."""
    params = {"api_key": LINKEDIN_API_KEY, "job_id": job_id}
    response = requests.get(LINKEDIN_BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def extract_skills(job_description: str) -> List[str]:
    """Extract skills from job description using OpenAI."""
    prompt = f"""
    Extract the most relevant skills from the following job description. 
    Return the skills as a comma-separated list:
    {job_description}
    """
    response = openai_request(prompt)
    return [skill.strip() for skill in response.split(",") if skill.strip()]

def save_job_data(file_path: str, jobs: List[Dict]):
    """Save job data to a CSV file."""
    headers = ["job_position", "company_name", "seniority_level", "job_location", "skills"]
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)
        for job in jobs:
            writer.writerow([
                job["job_position"],
                job["company_name"],
                job["seniority_level"],
                job["job_location"],
                ", ".join(job["skills"]),
            ])

def main():
    logger.info("Starting job data extraction pipeline.")

    # Load previously queried job IDs
    queried_ids = load_queried_ids(QUERIED_IDS_FILE)

    # Parameters
    field = "data scientist"
    max_job_ids = 10000

    for location_id in LOCATION_IDS:
        logger.info(f"Processing jobs for location ID: {location_id}")

        page = 1
        total_job_ids = len(queried_ids)
        api_credits_remaining = True

        while page <= 100 and total_job_ids < max_job_ids and api_credits_remaining:
            try:
                # Fetch job listings for the current page
                logger.info(f"Fetching job listings for page {page}.")
                job_listings = fetch_job_listings(field, location_id, page)

                if not job_listings:
                    logger.info("No more job listings available.")
                    break

                new_jobs = []
                new_job_ids = []

                for job in job_listings:
                    job_id = job["job_id"]
                    if job_id in queried_ids:
                        continue

                    try:
                        logger.info(f"Fetching details for job ID: {job_id}")
                        job_details = fetch_job_details(job_id)

                        # Ensure job_details is a dictionary (not a list)
                        if isinstance(job_details, list) and len(job_details) > 0:
                            job_details = job_details[0]

                        if not isinstance(job_details, dict):
                            logger.error(f"Unexpected format for job details: {job_details}")
                            continue

                        # Extract skills using OpenAI
                        job_description = job_details.get("job_description", "")
                        skills = extract_skills(job_description)

                        # Append processed job data
                        new_jobs.append({
                            "job_position": job_details.get("job_position", "N/A"),
                            "company_name": job_details.get("company_name", "N/A"),
                            "seniority_level": job_details.get("Seniority_level", "N/A"),
                            "job_location": job_details.get("job_location", "N/A"),
                            "skills": skills,
                        })

                        new_job_ids.append(job_id)

                    except requests.exceptions.RequestException as e:
                        logger.error(f"API request failed for job ID {job_id}: {e}")
                        continue  # Skip to the next job in the loop
                    except Exception as e:
                        logger.error(f"An unexpected error occurred for job ID {job_id}: {e}")
                        continue  # Skip to the next job in the loop

                # Save new job data and queried job IDs
                if new_jobs:
                    save_job_data(OUTPUT_FILE, new_jobs)
                    save_queried_ids(QUERIED_IDS_FILE, new_job_ids)
                    queried_ids.update(new_job_ids)
                    total_job_ids += len(new_job_ids)
                    logger.info(f"Processed {len(new_job_ids)} new jobs. Total: {total_job_ids}.")

                # Increment page for the next API call
                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                break  # Stop fetching more pages for this location if there's an API error
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                break  # Stop fetching more pages for this location if there's an unexpected error

    logger.info("Job data extraction pipeline completed.")

if __name__ == "__main__":
    main()
