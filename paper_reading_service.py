import sys
import os
import json
import time
import threading
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from llm import LLMClient, TokenUsageTracker
from tools import _read_text, load_prompt, pdf_to_md, _fix_json_escapes

# Import shared components from rebuttal_service
from rebuttal_service import LogCollector, ProcessStatus, get_llm_client

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_BASE_DIR = os.path.join(_CURRENT_DIR, "gradio_uploads")

os.makedirs(SESSIONS_BASE_DIR, exist_ok=True)


@dataclass
class PaperReadingSession:
    """Session state for paper reading workflow"""
    session_id: str
    pdf_path: str = ""
    md_path: str = ""
    research_field_md_path: str = ""
    research_field_text: str = ""  # Text read from MD file
    
    agent1_output: Dict = field(default_factory=dict)  # JSON object
    agent2_output: Dict = field(default_factory=dict)  # JSON object
    agent3_outputs: List[str] = field(default_factory=list)  # Array of analysis strings
    agent4_outputs: List[str] = field(default_factory=list)  # Array of application analysis strings
    agent5_output: Dict = field(default_factory=dict)  # JSON object
    
    session_dir: str = ""
    log_dir: str = ""
    log_collector: Optional[LogCollector] = None
    overall_status: ProcessStatus = ProcessStatus.NOT_STARTED
    progress_message: str = ""


def _parse_json_output(text: str, agent_name: str) -> Dict:
    """Parse JSON output from Agent response"""
    # Extract JSON part
    json_start = text.find('{')
    json_end = text.rfind('}') + 1
    
    if json_start == -1 or json_end <= json_start:
        raise ValueError(f"{agent_name}: No JSON found in output")
    
    json_str = _fix_json_escapes(text[json_start:json_end])
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"{agent_name}: JSON parsing failed: {e}")


class PaperReadingAgent1:
    """Extract core motivation and rough innovations"""
    def __init__(self, paper_file_path: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_file_path = paper_file_path
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self, paper_text: str) -> str:
        instructions = load_prompt("paper_reading_agent1.yaml")
        return f"{instructions}\n\n{paper_text}\n\n"
    
    def run(self) -> str:
        paper_text = _read_text(self.paper_file_path)
        model_input = self._build_context(paper_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent1_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="PaperReadingAgent1",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent1_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class PaperReadingAgent2:
    """Refine and double-check paper summary, innovations, and keywords"""
    def __init__(self, paper_file_path: str, agent1_output: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_file_path = paper_file_path
        self.agent1_output = agent1_output
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self, paper_text: str) -> str:
        instructions = load_prompt("paper_reading_agent2.yaml")
        return (
            f"{instructions}\n\n"
            f"[paper original text]\n\n{paper_text}\n\n"
            f"[Agent1 initial analysis]\n\n{self.agent1_output}\n\n"
        )
    
    def run(self) -> str:
        paper_text = _read_text(self.paper_file_path)
        model_input = self._build_context(paper_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="PaperReadingAgent2",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class PaperReadingAgent3:
    """Analyze detailed implementation, function, rationale, and theoretical basis of a specific innovation"""
    def __init__(self, paper_file_path: str, innovation: str, temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.paper_file_path = paper_file_path
        self.innovation = innovation
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self, paper_text: str) -> str:
        instructions = load_prompt("paper_reading_agent3.yaml")
        return (
            f"{instructions}\n\n"
            f"[paper original text]\n\n{paper_text}\n\n"
            f"[innovation description]\n\n{self.innovation}\n\n"
        )
    
    def run(self) -> str:
        paper_text = _read_text(self.paper_file_path)
        model_input = self._build_context(paper_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent3_innovation{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"PaperReadingAgent3_innovation{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent3_innovation{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class PaperReadingAgent4:
    """Analyze application value of innovation in user's research field"""
    def __init__(self, agent3_output: str, research_field_text: str, innovation: str, 
                 temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.agent3_output = agent3_output
        self.research_field_text = research_field_text
        self.innovation = innovation
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("paper_reading_agent4.yaml")
        return (
            f"{instructions}\n\n"
            f"[innovation detailed analysis from Agent3]\n\n{self.agent3_output}\n\n"
            f"[user research field description]\n\n{self.research_field_text}\n\n"
            f"[innovation brief description]\n\n{self.innovation}\n\n"
        )
    
    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent4_innovation{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"PaperReadingAgent4_innovation{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent4_innovation{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class PaperReadingAgent5:
    """Evaluate paper writing quality, rating, and writing characteristics"""
    def __init__(self, paper_file_path: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_file_path = paper_file_path
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self, paper_text: str) -> str:
        instructions = load_prompt("paper_reading_agent5.yaml")
        return f"{instructions}\n\n{paper_text}\n\n"
    
    def run(self) -> str:
        paper_text = _read_text(self.paper_file_path)
        model_input = self._build_context(paper_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent5_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="PaperReadingAgent5",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent5_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class PaperReadingService:
    """Paper reading workflow service class
    
    Designed as an independent, reusable service that can be called by other workflows.
    """
    def __init__(self):
        self.sessions: Dict[str, PaperReadingSession] = {}
        self._lock = threading.Lock()
    
    def create_session(
        self, 
        session_id: str, 
        pdf_path: str, 
        research_field_md_path: str
    ) -> PaperReadingSession:
        """Create a new session"""
        session_dir = os.path.join(SESSIONS_BASE_DIR, session_id)
        logs_dir = os.path.join(session_dir, "logs")
        
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        log_collector = LogCollector()
        
        session = PaperReadingSession(
            session_id=session_id,
            pdf_path=pdf_path,
            research_field_md_path=research_field_md_path,
            session_dir=session_dir,
            log_dir=logs_dir,
            log_collector=log_collector,
        )
        
        with self._lock:
            self.sessions[session_id] = session
        
        log_collector.add(f"Session created: {session_id}")
        print(f"[PaperReading] Created session: {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[PaperReadingSession]:
        """Get session by ID"""
        with self._lock:
            return self.sessions.get(session_id)
    
    def run_workflow(
        self, 
        session_id: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """Execute the complete workflow
        
        Returns:
            {
                "agent1": {...},  # JSON object
                "agent2": {...},  # JSON object
                "agent3": [...],  # Array of strings
                "agent4": [...],  # Array of strings
                "agent5": {...}   # JSON object
            }
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.overall_status = ProcessStatus.PROCESSING
        
        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(msg)
            if progress_callback:
                progress_callback(msg)
            print(f"[PaperReading] {msg}")
        
        try:
            # 1. Read research field MD file
            update_progress("Reading research field description...")
            research_field_text = _read_text(session.research_field_md_path)
            session.research_field_text = research_field_text
            
            # 2. PDF to MD (reuse tools.pdf_to_md)
            update_progress("Converting PDF to Markdown...")
            md_path = pdf_to_md(session.pdf_path, session.session_dir)
            if not md_path:
                raise RuntimeError("PDF conversion failed")
            session.md_path = md_path
            
            # 3. Agent1
            update_progress("Agent1: Extracting core motivation and innovations...")
            agent1 = PaperReadingAgent1(md_path, log_dir=session.log_dir)
            agent1_output_str = agent1.run()
            session.agent1_output = _parse_json_output(agent1_output_str, "Agent1")
            
            # 4. Agent2
            update_progress("Agent2: Refining summary and innovations...")
            agent2 = PaperReadingAgent2(md_path, agent1_output_str, log_dir=session.log_dir)
            agent2_output_str = agent2.run()
            session.agent2_output = _parse_json_output(agent2_output_str, "Agent2")
            
            # 5. Agent3 loop
            innovations = session.agent2_output.get("innovations", [])
            session.agent3_outputs = []
            for idx, innovation in enumerate(innovations):
                update_progress(f"Agent3: Analyzing innovation {idx+1}/{len(innovations)}...")
                agent3 = PaperReadingAgent3(md_path, innovation, num=idx+1, log_dir=session.log_dir)
                output = agent3.run()
                session.agent3_outputs.append(output)
            
            # 6. Agent4 loop
            session.agent4_outputs = []
            for idx, (innovation, agent3_output) in enumerate(zip(innovations, session.agent3_outputs)):
                update_progress(f"Agent4: Analyzing application value {idx+1}/{len(innovations)}...")
                agent4 = PaperReadingAgent4(agent3_output, research_field_text, innovation, 
                                           num=idx+1, log_dir=session.log_dir)
                output = agent4.run()
                session.agent4_outputs.append(output)
            
            # 7. Agent5
            update_progress("Agent5: Evaluating writing quality...")
            agent5 = PaperReadingAgent5(md_path, log_dir=session.log_dir)
            agent5_output_str = agent5.run()
            session.agent5_output = _parse_json_output(agent5_output_str, "Agent5")
            
            update_progress("Workflow complete!")
            session.overall_status = ProcessStatus.COMPLETED
            
            return {
                "agent1": session.agent1_output,
                "agent2": session.agent2_output,
                "agent3": session.agent3_outputs,
                "agent4": session.agent4_outputs,
                "agent5": session.agent5_output
            }
            
        except Exception as e:
            session.overall_status = ProcessStatus.ERROR
            session.progress_message = f"Error: {str(e)}"
            update_progress(f"Workflow failed: {str(e)}")
            raise


# Global service instance
paper_reading_service = PaperReadingService()
