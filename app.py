import streamlit as st
import os
import json
import sys
import io
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pptx import Presentation

# ==========================
# 1. UI & LOGGING SETUP
# ==========================
st.set_page_config(page_title="Harvard Publication Crew Pro", layout="wide")

# Custom CSS for the log window
st.markdown("""
    <style>
    .log-container {
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', Courier, monospace;
        height: 300px;
        overflow-y: scroll;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Harvard Research Crew (With Live Logs)")

# --- Sidebar: Load Balancing ---
def render_agent_config(agent_label, key_prefix):
    with st.sidebar.expander(f"⚙️ {agent_label}", expanded=False):
        provider = st.selectbox("Provider", ["Google Gemini", "Groq", "OpenRouter"], key=f"{key_prefix}_p")
        api_key = st.text_input("API Key", type="password", key=f"{key_prefix}_k")
        
        default_model = "gemini-2.5-flash"
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
            
            prs.save("harvard_output.pptx")
            return "SUCCESS: File 'harvard_output.pptx' created."
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

topic = st.text_area("Research Topic", placeholder="Enter your academic query...")

if st.button("🚀 Start Publication Flow"):
    llm_res = create_crew_llm(res_config)
    llm_wri = create_crew_llm(wri_config)
    llm_pre = create_crew_llm(pre_config)

    if not (llm_res and llm_wri and llm_pre):
        st.error("Missing configuration in the sidebar.")
    else:
        # 🟢 Create the Log Window
        st.subheader("🕵️ Agent Thought Process (Live Logs)")
        log_window = st.empty()
        log_output = io.StringIO()

        # Custom class to capture stdout and update streamlit
        class StreamToStreamlit:
            def __init__(self, storage, display):
                self.storage = storage
                self.display = display
                self.buffer = ""

            def write(self, data):
                self.buffer += data
                # Update the UI
                self.display.markdown(f'<div class="log-container">{self.buffer.replace("\n", "<br>")}</div>', unsafe_allow_html=True)
            
            def flush(self):
                pass

        # Redirect stdout
        sys.stdout = StreamToStreamlit(log_output, log_window)

        try:
            # Agents & Tasks
            researcher = Agent(role='Researcher', goal=f'Deep dive into {topic}', backstory="Harvard Prof.", llm=llm_res, verbose=True)
            writer = Agent(role='Writer', goal='Create academic paper.', backstory="Journal Editor.", llm=llm_wri, verbose=True)
            # ✅ FIXED: Explicitly added backstory
            presenter = Agent(role='Presentation Expert', goal='Convert paper into a JSON structure for PowerPoint.', backstory='You are a world-class visual storyteller and slide designer for Harvard professors.', tools=[PowerPointCreatorTool()], llm=llm_pre,verbose=True)
            t1 = Task(description=f"Research {topic}.", expected_output="Summary.", agent=researcher)
            t2 = Task(description="Write paper.", expected_output="Paper.", agent=writer, context=[t1])
            t3 = Task(description="Create PPT.", expected_output="Confirmation.", agent=presenter, context=[t2])

            crew = Crew(agents=[researcher, writer, presenter], tasks=[t1, t2, t3], process=Process.sequential)
            
            final_result = crew.kickoff()

            # Restore stdout
            sys.stdout = sys.__stdout__

            st.success("✅ Workflow Complete!")
            
            # Show Results
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
