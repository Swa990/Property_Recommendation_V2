# from planner_agent import planner_agent
# from research_agent import research_worker, dispatch_workers, aggregate_results

# from typing import TypedDict, Annotated
# import operator

# from langgraph.graph import StateGraph, START, END


# class MainState(TypedDict, total=False):
#     user_query: str
#     planner_output: dict
#     task: dict
#     worker_findings: Annotated[list[dict], operator.add]
#     aggregated_research: dict


# builder = StateGraph(MainState)

# builder.add_node("planner", planner_agent)
# builder.add_node("research_worker", research_worker)
# builder.add_node("aggregate_results", aggregate_results)

# builder.add_edge(START, "planner")

# builder.add_conditional_edges("planner", dispatch_workers, ["research_worker"])

# builder.add_edge("research_worker", "aggregate_results")
# builder.add_edge("aggregate_results", END)

# app = builder.compile()


# def run_property_research(query: str):
#     return app.invoke({
#         "user_query": query,
#         "worker_findings": []
#     })


# if __name__ == "__main__":
#     result = run_property_research("Find 2 BHK flats in Hinjewadi and Wakad under 75 lakh")
#     print(result["aggregated_research"])




from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from planner_agent import planner_agent
from research_agent import research_worker, dispatch_workers, aggregate_results
from evaluator_agent import evaluator_node, route_after_evaluation, research_more_node, finish_node


class MainState(TypedDict, total=False):
    user_query: str
    planner_output: dict
    task: dict
    worker_findings: Annotated[list[dict], operator.add]
    aggregated_research: dict
    requirements: dict
    evaluation: dict
    retry_count: int
    max_retries: int
    final_result: dict


def extract_requirements(state: MainState):
    return {"requirements": state["planner_output"]["requirements"]}


builder = StateGraph(MainState)

builder.add_node("planner", planner_agent)
builder.add_node("extract_requirements", extract_requirements)
builder.add_node("research_worker", research_worker)
builder.add_node("aggregate_results", aggregate_results)
builder.add_node("evaluator", evaluator_node)
builder.add_node("research_more", research_more_node)
builder.add_node("finish", finish_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "extract_requirements")

builder.add_conditional_edges("extract_requirements", dispatch_workers, ["research_worker"])

builder.add_edge("research_worker", "aggregate_results")
builder.add_edge("aggregate_results", "evaluator")

builder.add_conditional_edges("evaluator", route_after_evaluation, {
    "research_more": "research_more",
    "finish": "finish"
})

builder.add_conditional_edges("research_more", dispatch_workers, ["research_worker"])

builder.add_edge("finish", END)

app = builder.compile()


def run_property_research(query: str):
    return app.invoke({
        "user_query": query,
        "worker_findings": [],
        "retry_count": 0,
        "max_retries": 2
    })


if __name__ == "__main__":
    result = run_property_research("Find 2 BHK flats in Hinjewadi and Wakad under 75 lakh")
    print(result["final_result"])


# from strands import Agent
# from strands.models import BedrockModel

# model = BedrockModel(
#     model_id="in.anthropic.claude-haiku-4-5-20251001-v1:0",
#     region_name="ap-south-1"
# )

# agent = Agent(
#     model=model,
#     system_prompt="You are a helpful AI assistant."
# )

# response = agent("Explain AWS Strands in two sentences.")

# print(response)
