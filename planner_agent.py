
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from research_agent import MainState

load_dotenv()


# -----------------------------
# 1. LLM Configuration
# -----------------------------

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


# -----------------------------
# 2. Structured Output
# -----------------------------

# class PropertyRequirements(BaseModel):
#     property_type: str
#     bhk: Optional[str]
#     city: str
#     preferred_areas: list[str]
#     budget_min: Optional[int]
#     budget_max: Optional[int]
#     amenities: list[str]
#     connectivity_preferences: list[str]
#     other_preferences: list[str]


class PropertyRequirements(BaseModel):
    property_type: Optional[str] = None
    bhk: Optional[str] = None
    city: Optional[str] = None
    preferred_areas: list[str] = []
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    amenities: list[str] = []
    connectivity_preferences: list[str] = []
    other_preferences: list[str] = []

class ResearchTask(BaseModel):
    area: Optional[str] = None
    objective: str
    information_to_find: list[str]
    initial_search_queries: list[str]
    priority: str


class PlannerOutput(BaseModel):
    intent: str
    requirements: PropertyRequirements
    property_fields_to_collect: list[str]
    research_tasks: list[ResearchTask]
    research_strategy: str
    evaluation_criteria: list[str]


planner_llm = llm.with_structured_output(PlannerOutput)


# -----------------------------
# 3. Planner Agent
# -----------------------------

def planner_agent(state: MainState):

    # prompt = f"""
    # You are a real estate research planning agent.

    # Understand the customer's request and create a structured
    # research plan for the next Research Agent.

    # Customer Query:
    # {state['user_query']}

    # Your responsibilities:

    # 1. Identify the customer's intent.
    # 2. Extract all property requirements.
    # 3. Identify preferred locations separately.
    # 4. Identify mandatory and optional requirements.
    # 5. Create 3 research tasks based on locations and other
    #    independent research requirements.
    # 6. For every task, specify:
    #    - Research objective
    #    - Information to find
    #    - Initial Tavily search query suggestions
    #    - Priority
    # 7. Specify which fields the Research Agent must collect
    #    for every property.
    # 8. Define evaluation criteria for the Evaluator Agent.

    # Important instructions:

    # - Do not invent missing customer requirements.
    # - Use null for unknown budget limits or BHK.
    # - Do not create unnecessary duplicate tasks.
    # - Create separate property discovery tasks for
    #   different preferred locations.
    # - Include connectivity research when requested.
    # - Do not perform actual web searches.
    # - The Research Agent will decide whether additional
    #   searches are needed after examining search results.

    # Return a complete structured research plan.
    # """


    prompt = f"""
You are a real estate research planning agent.

Understand the customer's request and create a structured
research plan for the Research Agent.

Customer Query:
{state['user_query']}

Your responsibilities:

1. Identify the customer's intent.
2. Extract all explicitly stated property requirements.
3. Identify preferred locations separately.
4. Distinguish mandatory requirements from optional preferences.
5. Create up to 3 relevant research tasks based on locations
   and independent research objectives.
6. For every task, specify:
   - Research objective
   - Information to find
   - Initial Tavily search query suggestions
   - Priority
7. Specify which fields the Research Agent must collect
   for every property.
8. Define evaluation criteria for the Evaluator Agent.

Important instructions:

- Never invent or assume customer requirements.
- Use null for unspecified BHK, property type, or budget limits.
- Use empty lists for unspecified amenities, connectivity,
  and other preferences.
- If BHK is not specified, search across available BHK configurations.
- If property type is not specified, search across relevant
  residential property types.
- Do not apply filters for criteria the customer has not requested.
- Create separate property discovery tasks for different
  preferred locations.
- Include connectivity research only when explicitly requested.
- Avoid duplicate or unnecessary research tasks.
- Do not perform actual web searches.
- The Research Agent executes the tasks, and the Evaluator
  checks results against the original customer requirements.

Return a complete structured research plan.
"""


    result = planner_llm.invoke(prompt)

    output = result.model_dump()

    for i, task in enumerate(output["research_tasks"], start=1):
        task["task_id"] = f"T{i}"

    return {"planner_output": output}
