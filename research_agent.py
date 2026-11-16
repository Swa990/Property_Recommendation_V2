
import os
import operator
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send

load_dotenv()

llm = ChatOpenAI(
    model="Qwen/Qwen3.5-4B",
    base_url="https://prod-vllm-2.kriyam.ai/v1",
    api_key=os.getenv("VLM_API_TOKEN"),
    temperature=0,
    timeout=120,
    max_retries=2,
    max_tokens=8192,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

tavily = TavilySearch(max_results=3, search_depth="basic")


# -----------------------------
# 1. Structured Output
# -----------------------------

class PropertyFinding(BaseModel):
    property_name: Optional[str] = None
    location: Optional[str] = None
    price: Optional[str] = None
    bhk: Optional[str] = None
    area_sqft: Optional[str] = None
    amenities: list[str] = Field(default_factory=list)
    connectivity: list[str] = Field(default_factory=list)
    availability: Optional[str] = None
    source_url: Optional[str] = None
    evidence: Optional[str] = None
    missing_information: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_null_lists(cls, values):
        if isinstance(values, dict):
            for field in ["amenities", "connectivity", "missing_information"]:
                if values.get(field) is None:
                    values[field] = []
        return values


class ResearchOutput(BaseModel):
    properties: list[PropertyFinding] = Field(default_factory=list)
    location_information: list[str] = Field(default_factory=list)
    important_findings: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


# -----------------------------
# 2. LangGraph State
# -----------------------------

class MainState(TypedDict, total=False):
    user_query: str
    planner_output: dict
    task: dict
    worker_findings: Annotated[list[dict], operator.add]
    aggregated_research: dict


# -----------------------------
# 3. Tavily Search
# -----------------------------

def tavily_search(query: str) -> list[dict]:

    response = tavily.invoke({"query": query})

    results = response.get("results", []) if isinstance(response, dict) else response

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score")
        }
        for item in results or []
    ]


# -----------------------------
# 4. Research Worker
# -----------------------------

def research_worker(state: MainState) -> dict:

    task = state["task"]
    requirements = state["planner_output"]["requirements"]

    queries = task.get("initial_search_queries", [])[:5]

    raw_results = []

    for query in queries:
        results = tavily_search(query)
        raw_results.extend(results)

    unique_results = {
        item["url"]: item
        for item in raw_results
        if item.get("url")
    }

    raw_results = list(unique_results.values())

    raw_results = [
        {
            "title": r["title"],
            "url": r["url"],
            "content": r["content"][:1000]
        }
        for r in raw_results
    ]

    if not raw_results:
        return {
            "worker_findings": [{
                "task_id": task["task_id"],
                "objective": task["objective"],
                "properties": [],
                "missing_information": ["No search results retrieved"],
                "search_queries_used": queries
            }]
        }

    extractor = llm.with_structured_output(ResearchOutput)

    extracted = extractor.invoke([
        SystemMessage(content="""
You are a real estate research agent.

Extract top 3 property information only from the provided search results.

Rules:
1. Never invent property information.
2. Use null for missing scalar values.
3. For amenities, connectivity and missing_information,
   ALWAYS return an array.
4. If no amenities are mentioned, return [].
5. If no connectivity information is mentioned, return [].
6. If no missing information exists, return [].
7. Preserve original source URLs.
8. Distinguish verified information from unverified claims.
"""),
        HumanMessage(content=f"""
        CUSTOMER REQUIREMENTS:
        {requirements}

        CURRENT RESEARCH TASK:
        {task}

        TAVILY SEARCH RESULTS:
        {raw_results}

        Extract the relevant findings for this task.
        """)
    ])

    return {
        "worker_findings": [{
            "task_id": task["task_id"],
            "objective": task["objective"],
            "area": task.get("area"),
            "properties": [
                item.model_dump()
                for item in extracted.properties
            ],
            "location_information": extracted.location_information,
            "important_findings": extracted.important_findings,
            "missing_information": extracted.missing_information,
            "search_queries_used": queries
        }]
    }


# -----------------------------
# 5. Fan-out
# -----------------------------

def dispatch_workers(state: MainState):

    tasks = state["planner_output"]["research_tasks"]

    return [
        Send(
            "research_worker",
            {
                "user_query": state["user_query"],
                "planner_output": state["planner_output"],
                "task": task
            }
        )
        for task in tasks
    ]


# -----------------------------
# 6. Aggregate Worker Results
# -----------------------------

def aggregate_results(state: MainState) -> dict:

    worker_findings = state.get("worker_findings", [])

    all_properties = {}

    for worker in worker_findings:
        for prop in worker.get("properties", []):

            url = prop.get("source_url")

            if url:
                all_properties[url] = prop

    return {
        "aggregated_research": {
            "user_query": state["user_query"],
            "planner_output": state["planner_output"],
            "total_workers": len(worker_findings),
            "worker_findings": worker_findings,
            "all_properties": list(all_properties.values())
        }
    }
