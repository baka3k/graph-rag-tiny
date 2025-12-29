import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from environment variables
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_key = os.getenv("OPENAI_API_KEY")

print("Loaded environment variables")
print("qdrant_key:", qdrant_key)
print("qdrant_url:", qdrant_url)
print("neo4j_uri:", neo4j_uri)
print("neo4j_username:", neo4j_username)
print("neo4j_password:", neo4j_password)
print("openai_key:", openai_key)

print("Environment variables loaded successfully")
