import os  # Initialize Neo4j driver
import uuid
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from enviroment_loader import (
    neo4j_password,
    neo4j_uri,
    neo4j_username,
    qdrant_key,
    qdrant_url,
)

neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# Initialize Qdrant client
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
