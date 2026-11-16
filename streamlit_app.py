
import streamlit as st
from main import app

st.set_page_config(page_title="Property Finder", page_icon="🏠", layout="wide")

st.title("🏠 Property Finder")
st.caption("AI-powered property research and recommendations")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.json(message["content"])

def render_properties(properties, title):
    st.subheader(title)

    if not properties:
        st.info("No properties found.")
        return

    for i, prop in enumerate(properties, 1):
        if isinstance(prop, str):
            st.markdown(f"**{i}. {prop}**")
            continue

        with st.container(border=True):
            st.markdown(f"**{i}. {prop.get('property_name', 'Property')}**")

            cols = st.columns(3)
            cols[0].write(f"**Price:** {prop.get('price', 'Not available')}")
            cols[1].write(f"**BHK:** {prop.get('bhk', 'Not available')}")
            cols[2].write(f"**Location:** {prop.get('location', 'Not available')}")

            if prop.get("reason"):
                st.caption(prop["reason"])

            if prop.get("source_url"):
                st.markdown(f"[View listing]({prop['source_url']})")

def render_result(result):
    st.markdown("## Research Results")

    if result.get("overall_assessment"):
        st.info(result["overall_assessment"])

    render_properties(result.get("matched_properties", []), "✅ Recommended Properties")
    render_properties(result.get("partial_match_properties", []), "⚠️ Partially Verified")
    
    rejected = result.get("rejected_properties", [])
    if rejected:
        with st.expander("Rejected / Ineligible Properties"):
            render_properties(rejected, "Rejected Properties")

    if result.get("retry_limit_reached"):
        st.warning("Research retry limit reached. Some information may remain unverified.")

    if not any(key in result for key in ["matched_properties", "partial_match_properties", "rejected_properties"]):
        st.json(result)


def run_research(query):
    inputs = {
        "user_query": query,
        "worker_findings": [],
        "retry_count": 0,
        "max_retries": 2
    }

    final_result = None
    worker_count = 0
    retry_count = 0

    with st.status("Understanding your requirements...", expanded=True) as status:

        status.write("🧠 Analyzing your property requirements...")

        try:
            for event in app.stream(inputs, stream_mode="updates"):

                for node, output in event.items():

                    if node == "planner":
                        status.update(label="Requirements understood", state="running")
                        status.write("📋 Identifying budget, BHK and preferred locations.")

                    elif node == "extract_requirements":
                        status.write("🔎 Preparing property research tasks.")

                    elif node == "research_worker":
                        worker_count += 1
                        status.update(label="Researching property listings...", state="running")
                        status.write(f"🌐 Research task {worker_count} completed.")

                    elif node == "aggregate_results":
                        status.update(label="Organizing property listings...", state="running")
                        status.write("🔄 Combining research findings and removing duplicates.")

                    elif node == "evaluator":
                        status.update(label="Evaluating properties...", state="running")
                        status.write("⚖️ Checking properties against your requirements.")

                    elif node == "research_more":
                        retry_count += 1
                        status.update(label="Looking for missing information...", state="running")
                        status.write(f"🔍 Additional research attempt {retry_count}.")

                    elif node == "finish":
                        final_result = output.get("final_result")

            if final_result is None:
                status.update(label="Research completed, but no final result was returned.", state="error")
                return None

            status.update(label="Property research completed!", state="complete", expanded=False)
            return final_result

        except Exception as e:
            status.update(label="Something went wrong during research.", state="error")
            st.exception(e)
            return None


query = st.chat_input("Describe the property you are looking for...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = run_research(query)

        if result:
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            st.error("Unable to complete the property research.")
