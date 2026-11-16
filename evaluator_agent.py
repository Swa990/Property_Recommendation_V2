
import os
from typing import TypedDict, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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


# class PropertyEvaluation(BaseModel):
#     property_name: str
#     source_url: Optional[str] = None
#     status: Literal["MATCH", "PARTIAL_MATCH", "REJECT"]
#     budget_check: str
#     bhk_check: str
#     location_check: str
#     gym_check: str
#     covered_parking_check: str
#     connectivity_check: str
#     reason: str
#     missing_information: Optional[list[str]] = None

class PropertyEvaluation(BaseModel):
    property_name: str
    source_url: Optional[str] = None
    price: Optional[str] = None
    bhk: Optional[str] = None
    location: Optional[str] = None
    area_sqft: Optional[str] = None
    status: Literal["MATCH", "PARTIAL_MATCH", "REJECT"]
    budget_check: str
    bhk_check: str
    location_check: str
    gym_check: str
    covered_parking_check: str
    connectivity_check: str
    reason: str
    missing_information: Optional[list[str]] = None


class FollowUpTask(BaseModel):
    area: Optional[str] = None
    objective: str
    information_to_find: Optional[list[str]] = None
    initial_search_queries: Optional[list[str]] = None
    priority: str = "High"


class EvaluationOutput(BaseModel):
    decision: Literal["ENOUGH_INFORMATION", "RESEARCH_MORE"]
    property_evaluations: Optional[list[PropertyEvaluation]] = None
    relevant_properties: Optional[list[str]] = None
    rejected_properties: Optional[list[str]] = None
    missing_information: Optional[list[str]] = None
    followup_tasks: Optional[list[FollowUpTask]] = None
    overall_assessment: str


class MainState(TypedDict, total=False):
    user_query: str
    planner_output: dict
    requirements: dict
    aggregated_research: dict
    evaluation: dict
    retry_count: int
    max_retries: int
    final_result: dict


evaluator_llm = llm.with_structured_output(EvaluationOutput)


def evaluator_node(state: MainState) -> dict:

    requirements = state["requirements"]
    research = state["aggregated_research"]
    properties = research.get("all_properties", [])

    requested_amenities = requirements.get("amenities") or []
    requested_connectivity = requirements.get("connectivity_preferences") or []
    other_preferences = requirements.get("other_preferences") or []

#     prompt = f"""
# You are a strict real estate property evaluation agent.

# CUSTOMER QUERY:
# {state['user_query']}

# CUSTOMER REQUIREMENTS:
# {requirements}

# EXPLICITLY REQUESTED AMENITIES:
# {requested_amenities}

# EXPLICITLY REQUESTED CONNECTIVITY:
# {requested_connectivity}

# OTHER PREFERENCES:
# {other_preferences}

# RESEARCHED PROPERTIES:
# {properties}

# YOUR RESPONSIBILITY:

# Evaluate every property strictly against the customer's explicitly
# requested requirements. Use only the supplied research evidence.
# Never invent or assume property information.

# EVALUATION CRITERIA:

# 1. BUDGET:
#    Verify whether the property's price satisfies the customer's
#    budget. Reject properties that clearly exceed the maximum budget.
#    If a price range overlaps the budget but does not establish an
#    eligible specific unit price, classify it as UNKNOWN.

# 2. BHK:
#    Verify the explicitly requested BHK configuration.
#    Reject properties with a clearly different configuration.
#    Ambiguous configurations must be classified as UNKNOWN.

# 3. LOCATION:
#    Verify whether the property is located in one of the customer's
#    preferred areas. Missing or ambiguous location is UNKNOWN.

# 4. AMENITIES, CONNECTIVITY AND OTHER PREFERENCES:
#    Evaluate only preferences explicitly requested by the customer.
#    Unrequested criteria must be marked NOT_REQUESTED and must not
#    affect the property's evaluation status.

# EVALUATION RULES:

# 1. Missing information for a requested criterion must be classified
#    as UNKNOWN, never assumed to be satisfied or failed.

# 2. Missing critical eligibility information (price, BHK or location)
#    makes a property ineligible for recommendation.

# 3. Clearly distinguish between a failed requirement and information
#    that is unavailable in the research evidence.

# 4. Reserved parking does not automatically mean covered parking.
#    Generic parking does not automatically mean covered parking.

# 5. When property-specific connectivity is requested, general area
#    connectivity is not sufficient evidence.

# 6. Combine evidence from multiple research records referring to
#    the same property before evaluating it. Preserve conflicting
#    values and source information rather than arbitrarily selecting
#    one.

# PROPERTY INFORMATION EXTRACTION:

# For every property evaluation, populate:

# - property_name: Property/project name.
# - price: Exact price or published price range.
# - bhk: Explicit BHK configuration.
# - location: Property location/locality.
# - area_sqft: Property area, if available.

# Extract these fields directly from the research records.
# Preserve price ranges and use null only when information is
# genuinely unavailable. Never put available structured information
# only inside the reason field.

# CLASSIFICATION:

# MATCH:
# All mandatory requirements are verified and satisfied.
# All explicitly requested non-mandatory preferences are also
# verified and satisfied.

# PARTIAL_MATCH:
# All mandatory requirements are verified and satisfied,
# but one or more explicitly requested non-mandatory preferences
# remain unverified or unsatisfied.

# REJECT:
# Use REJECT when:
# - A mandatory requirement clearly fails.
# - Critical eligibility information (price, BHK or location)
#   is missing or cannot be verified.

# Properties classified as REJECT must not appear in the
# recommended properties list.

# FOLLOW-UP RESEARCH:

# Generate targeted follow-up tasks only when additional research
# could resolve missing information about potentially relevant
# properties.

# Rules:
# 1. Research only missing information related to explicitly
#    requested criteria.
# 2. Do not research unrequested amenities or connectivity.
# 3. Do not create tasks for properties that clearly fail a
#    mandatory requirement.
# 4. Properties rejected only because critical information is
#    unavailable may receive follow-up tasks if that information
#    can reasonably be verified.
# 5. Each task must include:
#    - area
#    - objective
#    - information_to_find
#    - initial_search_queries
#    - priority
# 6. Avoid duplicate follow-up tasks.

# RESEARCH DECISION:

# RESEARCH_MORE:
# Important information about requested criteria is missing,
# and targeted research could reasonably resolve it.

# ENOUGH_INFORMATION:
# Sufficient information has been collected, or additional
# research is unlikely to improve the results.

# OUTPUT LIMITS:

# - Maximum 20 property evaluations.
# - Maximum 5 follow-up tasks.
# - Keep reasons and explanations concise.
# - Return only the required structured evaluation.

# CURRENT RETRY COUNT: {state.get("retry_count", 0)}
# MAXIMUM ADDITIONAL RESEARCH ATTEMPTS: {state.get("max_retries", 2)}

# Return the structured evaluation.
# """



    prompt = f"""
You are a real estate property evaluation agent.

CUSTOMER QUERY:
{state['user_query']}

CUSTOMER REQUIREMENTS:
{requirements}

EXPLICITLY REQUESTED AMENITIES:
{requested_amenities}

EXPLICITLY REQUESTED CONNECTIVITY:
{requested_connectivity}

OTHER PREFERENCES:
{other_preferences}

RESEARCHED PROPERTIES:
{properties}

YOUR RESPONSIBILITY:

Evaluate every property against only the customer's explicitly
requested requirements. Use the supplied research evidence.
Never invent or assume property information.

EVALUATION CRITERIA:

1. BUDGET:
   Evaluate only if the customer specified a budget.
   Reject properties whose verified price exceeds the budget.
   If a price range overlaps the budget but does not confirm
   an eligible specific unit, classify the budget as UNKNOWN.

2. BHK:
   Evaluate only if the customer explicitly requested a BHK.
   Reject a clearly different BHK configuration.
   If BHK is requested but missing or ambiguous, classify it
   as UNKNOWN.
   If BHK is not requested, mark bhk_check as NOT_REQUESTED.

3. PROPERTY TYPE:
   Evaluate only if the customer explicitly requested a
   property type. Otherwise, do not filter by property type.

4. LOCATION:
   Evaluate the requested city and preferred areas.
   Missing or ambiguous location is UNKNOWN.

5. AMENITIES, CONNECTIVITY AND OTHER PREFERENCES:
   Evaluate only explicitly requested preferences.
   Mark unrequested criteria as NOT_REQUESTED.

EVALUATION RULES:

1. Never treat an unrequested criterion as mandatory.
2. Missing information for a requested criterion must be
   classified as UNKNOWN.
3. A property is ineligible for recommendation if a requested
   mandatory criterion fails or critical information required
   to verify that criterion is missing.
4. Clearly distinguish between a failed requirement and
   unavailable information.
5. Reserved parking does not automatically mean covered parking.
6. When property-specific connectivity is requested, general
   area connectivity is not sufficient evidence.
7. Combine evidence from records referring to the same property.
   Preserve conflicting values and source information.

PROPERTY INFORMATION EXTRACTION:

For every property evaluation, populate:

- property_name: Property/project name.
- source_url: Source listing URL.
- price: Exact price or published price range.
- bhk: Explicit BHK configuration.
- location: Property location/locality.
- area_sqft: Property area, if available.

Extract these fields from research evidence.
Preserve price ranges and use null when information is unavailable.
Never put available structured information only inside the reason.

CLASSIFICATION:

MATCH:
All explicitly requested mandatory requirements are verified
and satisfied. Unrequested criteria do not affect classification.

PARTIAL_MATCH:
All mandatory requirements are verified and satisfied, but
one or more explicitly requested non-mandatory preferences
remain unverified or unsatisfied.

REJECT:
A requested mandatory requirement clearly fails, or critical
information needed to verify a requested mandatory requirement
is missing.

Rejected properties must not appear in recommendations.

FOLLOW-UP RESEARCH:

Generate follow-up tasks only when additional research could
resolve missing information related to explicitly requested
requirements.

1. Do not research unrequested amenities, connectivity,
   BHK or property type.
2. Do not create tasks for properties that clearly fail
   a mandatory requirement.
3. Properties rejected only because requested critical
   information is unavailable may receive follow-up tasks.
4. Each task must include area, objective, information_to_find,
   initial_search_queries and priority.
5. Avoid duplicate follow-up tasks.

RESEARCH DECISION:

RESEARCH_MORE:
Requested information is missing and additional research
could reasonably resolve it.

ENOUGH_INFORMATION:
Sufficient information has been collected, or additional
research is unlikely to improve the results.

OUTPUT LIMITS:

- Maximum 20 property evaluations.
- Maximum 5 follow-up tasks.
- Keep reasons concise.
- Return only the required structured evaluation.

CURRENT RETRY COUNT: {state.get("retry_count", 0)}
MAXIMUM ADDITIONAL RESEARCH ATTEMPTS: {state.get("max_retries", 2)}

Return the structured evaluation.
"""


    result = evaluator_llm.invoke([
        SystemMessage(content="Evaluate properties strictly against explicitly requested customer requirements."),
        HumanMessage(content=prompt)
    ])

    output = result.model_dump()

    for key in [
        "property_evaluations",
        "relevant_properties",
        "rejected_properties",
        "missing_information",
        "followup_tasks"
    ]:
        output[key] = output.get(key) or []

    for prop in output["property_evaluations"]:
        prop["missing_information"] = prop.get("missing_information") or []

    for task in output["followup_tasks"]:
        task["information_to_find"] = task.get("information_to_find") or []
        task["initial_search_queries"] = task.get("initial_search_queries") or []

    return {"evaluation": output}


def route_after_evaluation(state: MainState):

    evaluation = state["evaluation"]

    should_research = (
        evaluation["decision"] == "RESEARCH_MORE"
        and bool(evaluation.get("followup_tasks"))
        and state.get("retry_count", 0) < state.get("max_retries", 2)
    )

    return "research_more" if should_research else "finish"


def research_more_node(state: MainState) -> dict:

    evaluation = state["evaluation"]
    retry_count = state.get("retry_count", 0) + 1

    research_tasks = []

    for i, task in enumerate(evaluation.get("followup_tasks", []), start=1):
        research_tasks.append({
            **task,
            "task_id": f"R{retry_count}_T{i}"
        })

    return {
        "planner_output": {
            "requirements": state["requirements"],
            "research_tasks": research_tasks
        },
        "retry_count": retry_count,
        "worker_findings": []
    }


def finish_node(state: MainState) -> dict:

    evaluation = state["evaluation"]
    evaluations = evaluation.get("property_evaluations", [])

    matched = [p for p in evaluations if p["status"] == "MATCH"]
    partial = [p for p in evaluations if p["status"] == "PARTIAL_MATCH"]
    rejected = [p for p in evaluations if p["status"] == "REJECT"]

    retry_count = state.get("retry_count", 0)

    return {
        "final_result": {
            "user_query": state["user_query"],
            "requirements": state["requirements"],
            "retry_count": retry_count,
            "retry_limit_reached": (
                retry_count >= state.get("max_retries", 2)
                and evaluation["decision"] == "RESEARCH_MORE"
            ),
            "matched_properties": matched,
            "partial_match_properties": partial,
            "rejected_properties": rejected,
            "missing_information": evaluation.get("missing_information", []),
            "overall_assessment": evaluation["overall_assessment"],
            "all_research": state["aggregated_research"]
        }
    }
