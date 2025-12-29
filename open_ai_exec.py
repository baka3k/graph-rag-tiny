import os
import uuid
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

client = OpenAI()


def openai_llm_parser(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """ You are a precise graph relationship extractor. Extract all
                    relationships from the text and format them as a JSON object
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity",
                             "target_node": "Related Entity",
                             "relationship": "Type of Relationship"},
                            ...more relationships...
                        ]
                    }
                    Include ALL relationships mentioned in the text, including
                    implicit ones. Be thorough and precise. """,
            },
            {"role": "user", "content": prompt},
        ],
    )

    return GraphComponents.model_validate_json(completion.choices[0].message.content)
