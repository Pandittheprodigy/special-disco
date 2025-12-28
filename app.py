import streamlit as st
import os
import json
import sys
import io
import logging
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun # Search Tool
from pptx import Presentation

# ==========================
# 1. UI & LOGGING SUPPRESSION
# ==========================
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

st.set_page_config(page_title="Harvard Publication Crew Pro", layout="wide")

st.markdown("""
    <style>
    .log-container {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        height: 350px;
        overflow-y: scroll;
        font-size: 0.85rem;
        border: 1px solid #444;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Harvard Research Crew (Real-Time 2025)")

# --- Sidebar: Load Balancing ---
def render_agent_config(agent_label, key_prefix):
    with st.sidebar.expander(f"⚙️ {agent_label}", expanded=False):
        provider = st.selectbox("Provider", ["Google Gemini", "Groq", "OpenRouter"], key=f"{key_prefix}_p")
        api_key = st.text_input("API Key", type="password", key=f"{key_prefix}_k")
        
        default_model = "gemini-2.0-flash" 
        if provider == "Groq": default_model = "llama-3.3-70b-versatile"
        elif provider == "OpenRouter": default_model = "meta-llama/llama-3.3-70b-instruct:free"
        
        model_name = st.text_input("Model Name", value=default_model, key=f"{key_prefix}_m")
        return provider, api_key, model_name

with st.sidebar:
    st.header("🔑 Multi-Key Config")
    res_config = render_agent_config("Researcher (Key 1)", "res")
    wri_config = render_agent_config("Writer (Key 2)", "wri")
    pre_config = render_agent_config("Presenter (Key 3)", "pre")

# ==========================
# 2. TOOLS & LLM FACTORY
# ==========================

# Custom Web Search Tool Wrapper
class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the internet for real-time 2025 data, news, and academic trends."

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

class PowerPointCreatorTool(BaseTool):
    name: str = "Create PowerPoint"
    description: str = "Creates a .pptx file from JSON input."

    def _run(self, slides_json: str) -> str:
        try:
            clean_json = slides_json.replace("```json", "").replace("```", "").strip()
            start = clean_json.find('{')
            end = clean_json.rfind('}') + 1
            data = json.loads(clean_json[start:end])
            
            prs = Presentation()
            for slide_info in data.get('slides', []):
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = slide_info.get('title', 'Harvard Research')
                if len(slide.placeholders) > 1:
                    tf = slide.placeholders[1].text_frame
                    for point in slide_info.get('content', []):
                        p = tf.add_paragraph()
                        p.text = str(point)
            
            output_path = "harvard_output.pptx"
            prs.save(output_path)
            return f"SUCCESS: File '{output_path}' created."
        except Exception as e:
            return f"Error: {str(e)}"

def create_crew_llm(config_tuple):
    provider, api_key, model_name = config_tuple
    if not api_key: return None
    prefix_map = {"Google Gemini": "gemini/", "Groq": "groq/", "OpenRouter": "openrouter/"}
    full_model = f"{prefix_map.get(provider, '')}{model_name}"
    return LLM(model=full_model, api_key=api_key, temperature=0.3)

# ==========================
# 3. MAIN APP LOGIC
# ==========================

topic = st.text_area("Research Topic", placeholder="e.g., Global Economic Trends in Q4 2025...")

if st.button("🚀 Start Search & Publication Flow"):
    llm_res = create_crew_llm(res_config)
    llm_wri = create_crew_llm(wri_config)
    llm_pre = create_crew_llm(pre_config)

    if not (llm_res and llm_wri and llm_pre):
        st.error("Please ensure all API keys are provided.")
    else:
        st.subheader("🕵️ Agent Thought Process (Live Logs)")
        log_window = st.empty()
        
        class StreamToStreamlit:
            def __init__(self, display):
                self.display = display
                self.buffer = ""

            def write(self, data):
                self.buffer += data
                self.display.markdown(f'<div class="log-container">{self.buffer.replace("\n", "<br>")}</div>', unsafe_allow_html=True)
            
            def flush(self):
                pass

        sys.stdout = StreamToStreamlit(log_window)

        try:
            # --- Agents ---
            researcher = Agent(
                role='Principal Researcher', 
                goal=f'Use web search to find the latest 2025 data on {topic}', 
                backstory="You are a data-driven Harvard researcher with live access to the web. You provide actual links and citations.", 
                tools=[WebSearchTool()], # Added Search Tool
                llm=llm_res, 
                verbose=True
            )
            
            writer = Agent(
                role='Lead Academic Writer', 
                goal='Synthesize the searched data into a formal paper.', 
                backstory="You are a chief editor who transforms raw web data into scholarly articles.", 
                llm=llm_wri, 
                verbose=True
            )
            
            presenter = Agent(
                role='Presentation Specialist', 
                goal='Create a PowerPoint deck from the final paper.', 
                backstory='You design presentations for high-stakes academic conferences.', 
                tools=[PowerPointCreatorTool()], 
                llm=llm_pre,
                verbose=True
            )

            # --- Tasks ---
            t1 = Task(description=f"Search for and synthesize the latest 2025 information on {topic}.", expected_output="A research report with real-world citations.", agent=researcher)
            t2 = Task(description="Write a Harvard-style paper based on the research report.", expected_output="A full markdown paper.", agent=writer, context=[t1])
            t3 = Task(description="Generate a 7-slide PPT deck using the tool.", expected_output="Confirmation of PPTX creation.", agent=presenter, context=[t2])

            crew = Crew(agents=[researcher, writer, presenter], tasks=[t1, t2, t3], process=Process.sequential)
            
            final_result = crew.kickoff()
            sys.stdout = sys.__stdout__

            st.success("✅ Research & Publication Complete!")
            
            tab1, tab2 = st.tabs(["📄 Final Paper", "📥 Downloads"])
            with tab1:
                st.markdown(t2.output.raw)
            with tab2:
                if os.path.exists("harvard_output.pptx"):
                    with open("harvard_output.pptx", "rb") as f:
                        st.download_button("Download PowerPoint", f, "presentation.pptx")

        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Execution Error: {str(e)}")
