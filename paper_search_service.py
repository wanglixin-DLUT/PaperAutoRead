import os
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Callable, Tuple

from arxiv import ArxivAgent
from rebuttal_service import LogCollector, ProcessStatus, get_llm_client
from tools import (
    _read_text,
    load_prompt,
    pdf_to_md,
    _fix_json_escapes,
    download_pdf_only,
    submit_llm_request,
)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_BASE_DIR = os.path.join(_CURRENT_DIR, "gradio_uploads")

os.makedirs(SESSIONS_BASE_DIR, exist_ok=True)


@dataclass
class PaperSearchSession:
    session_id: str
    research_md_path: str = ""
    research_text: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    arxiv_category: str = ""
    arxiv_custom: str = ""
    top_k: int = 30
    query_text: str = ""
    final_query: str = ""
    papers: List[Dict] = field(default_factory=list)
    selected_labels: List[str] = field(default_factory=list)
    downloaded_manifest_path: str = ""
    pdf_paths: Dict[str, str] = field(default_factory=dict)
    md_paths: Dict[str, str] = field(default_factory=dict)
    agent2_outputs: List[Dict] = field(default_factory=list)
    innovations: List[Dict] = field(default_factory=list)
    agent3_outputs: List[Dict] = field(default_factory=list)
    adaptable: List[Dict] = field(default_factory=list)
    not_adaptable: List[Dict] = field(default_factory=list)
    combination_output: Dict = field(default_factory=dict)
    session_dir: str = ""
    log_dir: str = ""
    log_collector: Optional[LogCollector] = None
    overall_status: ProcessStatus = ProcessStatus.NOT_STARTED
    progress_message: str = ""


def _parse_json_output(text: str, agent_name: str) -> Dict:
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError(f"{agent_name}: No JSON found in output")
    json_str = _fix_json_escapes(text[json_start:json_end])
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"{agent_name}: JSON parsing failed: {e}")


def _normalize_date(value: Optional[object]) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y%m%d")
    value_str = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(value_str, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return None


def _format_date_range(start_date: Optional[object], end_date: Optional[object]) -> Optional[str]:
    start_norm = _normalize_date(start_date)
    end_norm = _normalize_date(end_date)
    if not start_norm and not end_norm:
        return None
    if not start_norm:
        start_norm = "19000101"
    if not end_norm:
        end_norm = datetime.now().strftime("%Y%m%d")
    return f"submittedDate:[{start_norm}0000 TO {end_norm}2359]"


def _build_arxiv_query(base_query: str, category: str, custom: str, start_date: Optional[object], end_date: Optional[object]) -> str:
    clauses = []
    if base_query:
        normalized = base_query.strip()
        if any(token in normalized for token in ("abs:", "ti:", "au:", "cat:")):
            clauses.append(f"({normalized})")
        else:
            # 默认在摘要中检索，提升匹配覆盖面
            clauses.append(f"abs:({normalized})")
    if category and category != "全部":
        clauses.append(f"cat:{category}")
    if custom:
        custom_clause = custom.strip()
        if custom_clause:
            if "cat:" in custom_clause or "AND" in custom_clause or "OR" in custom_clause:
                clauses.append(custom_clause)
            else:
                clauses.append(f"cat:{custom_clause}")
    date_clause = _format_date_range(start_date, end_date)
    if date_clause:
        clauses.append(date_clause)
    if not clauses:
        return base_query
    return " AND ".join(clauses)


def _build_paper_label(paper: Dict) -> str:
    title = paper.get("title", "Unknown")
    arxiv_id = paper.get("arxiv_id", "N/A")
    return f"{title} ({arxiv_id})"


def _create_fallback_markdown(paper: Dict, output_dir: str, reason: str = "") -> str:
    title = paper.get("title", "Unknown Paper")
    abstract = paper.get("abstract", "No abstract available")
    arxiv_id = paper.get("arxiv_id", "N/A")
    pdf_url = paper.get("pdf_url", "")
    abs_url = paper.get("abs_url", "")
    authors = paper.get("authors", [])
    authors_str = ", ".join(authors) if authors else "Unknown"

    content = f"""# {title}

**arXiv ID**: {arxiv_id}  
**Authors**: {authors_str}  
**PDF URL**: {pdf_url}  
**Abstract URL**: {abs_url}  

---

**Note**: PDF download or conversion failed. Only metadata is available.
{f"**Error**: {reason}" if reason else ""}

---

## Abstract

{abstract}
"""

    safe_name = (arxiv_id or title).replace("/", "_").replace(":", "_")
    path = os.path.join(output_dir, f"{safe_name}.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class PaperSearchService:
    def __init__(self):
        self.sessions: Dict[str, PaperSearchSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        session_id: str,
        research_md_path: str,
        start_date: Optional[object],
        end_date: Optional[object],
        arxiv_category: str,
        arxiv_custom: str,
        top_k: int = 30,
    ) -> PaperSearchSession:
        session_dir = os.path.join(SESSIONS_BASE_DIR, session_id)
        logs_dir = os.path.join(session_dir, "logs")
        papers_dir = os.path.join(session_dir, "papers")
        md_dir = os.path.join(session_dir, "papers_md")

        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(papers_dir, exist_ok=True)
        os.makedirs(md_dir, exist_ok=True)

        log_collector = LogCollector()

        session = PaperSearchSession(
            session_id=session_id,
            research_md_path=research_md_path,
            start_date=start_date,
            end_date=end_date,
            arxiv_category=arxiv_category or "",
            arxiv_custom=arxiv_custom or "",
            top_k=top_k,
            session_dir=session_dir,
            log_dir=logs_dir,
            log_collector=log_collector,
        )

        with self._lock:
            self.sessions[session_id] = session

        log_collector.add(f"Session created: {session_id}")
        print(f"[PaperSearch] Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[PaperSearchSession]:
        with self._lock:
            return self.sessions.get(session_id)

    def run_query_agent(self, session_id: str) -> Dict:
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(msg)
            print(f"[PaperSearch] {msg}")

        update_progress("Agent1: 构造检索式...")
        research_text = _read_text(session.research_md_path)
        session.research_text = research_text

        instructions = load_prompt("paper_search_agent1.yaml")
        date_clause = _format_date_range(session.start_date, session.end_date) or "未指定"
        context = (
            f"{instructions}\n\n"
            f"[user research field description]\n{research_text}\n\n"
            f"[date range]\n{date_clause}\n\n"
            f"[arxiv category]\n{session.arxiv_category or '未指定'}\n\n"
            f"[arxiv custom category]\n{session.arxiv_custom or '未指定'}\n\n"
        )

        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON."
        output_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=context,
            enable_reasoning=True,
            temperature=0.4,
            agent_name="PaperSearchAgent1",
        )

        agent1_output = _parse_json_output(output_text, "Agent1")
        query_text = agent1_output.get("search_query", "").strip()
        session.query_text = query_text
        update_progress("Agent1 完成")
        return agent1_output

    def search_arxiv(self, session_id: str, query_text: str) -> Tuple[List[Dict], str]:
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(msg)
            print(f"[PaperSearch] {msg}")

        update_progress("开始检索 arXiv...")
        session.query_text = query_text.strip()
        final_query = _build_arxiv_query(
            base_query=session.query_text,
            category=session.arxiv_category,
            custom=session.arxiv_custom,
            start_date=session.start_date,
            end_date=session.end_date,
        )
        session.final_query = final_query

        agent = ArxivAgent(
            max_results=session.top_k,
            download_dir=os.path.join(session.session_dir, "papers"),
            md_dir=os.path.join(session.session_dir, "papers_md"),
        )
        papers = agent._search_arxiv(final_query, max_results=session.top_k * 2)
        session.papers = papers[: session.top_k]
        update_progress(f"检索完成，找到 {len(session.papers)} 篇论文")
        return session.papers, final_query

    def run_workflow(
        self,
        session_id: str,
        selected_labels: List[str],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict:
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
            print(f"[PaperSearch] {msg}")

        try:
            update_progress("Step2: 下载选定论文...")
            session.selected_labels = selected_labels
            label_to_paper = {_build_paper_label(p): p for p in session.papers}
            selected_papers = [label_to_paper[l] for l in selected_labels if l in label_to_paper]

            pdf_dir = os.path.join(session.session_dir, "papers")
            md_dir = os.path.join(session.session_dir, "papers_md")

            manifest = []
            session.pdf_paths = {}
            for paper in selected_papers:
                pdf_path = download_pdf_only(paper, pdf_dir)
                arxiv_id = paper.get("arxiv_id") or paper.get("title", "unknown")
                if pdf_path:
                    session.pdf_paths[arxiv_id] = pdf_path
                manifest.append({
                    "title": paper.get("title"),
                    "arxiv_id": paper.get("arxiv_id"),
                    "pdf_path": pdf_path,
                    "abs_url": paper.get("abs_url"),
                })

            manifest_path = os.path.join(session.session_dir, "downloaded_papers.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            session.downloaded_manifest_path = manifest_path

            update_progress("Step3: 转换 PDF → Markdown，并并行触发创新点提取...")
            agent2_requests = []
            session.md_paths = {}
            client = get_llm_client()

            for idx, paper in enumerate(selected_papers, 1):
                arxiv_id = paper.get("arxiv_id") or paper.get("title", "unknown")
                pdf_path = session.pdf_paths.get(arxiv_id)
                if pdf_path:
                    md_path = pdf_to_md(pdf_path, md_dir)
                else:
                    md_path = None
                if not md_path:
                    md_path = _create_fallback_markdown(paper, md_dir, reason="PDF 下载或转换失败")
                session.md_paths[arxiv_id] = md_path

                paper_text = _read_text(md_path)
                instructions = load_prompt("paper_search_agent2.yaml")
                context = (
                    f"{instructions}\n\n"
                    f"[paper metadata]\n{json.dumps(paper, ensure_ascii=False)}\n\n"
                    f"[paper markdown]\n{paper_text}\n\n"
                )
                request = submit_llm_request(
                    client=client,
                    instructions="Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON.",
                    input_text=context,
                    enable_reasoning=True,
                    temperature=0.4,
                    agent_name=f"PaperSearchAgent2_{idx}",
                )
                agent2_requests.append((paper, request))

            update_progress("Step4: 等待创新点提取完成...")
            session.agent2_outputs = []
            session.innovations = []
            for paper, request in agent2_requests:
                result, error = request.wait()
                if error or not result:
                    update_progress(f"Agent2 失败：{paper.get('title', 'Unknown')}")
                    continue
                output_text = result[0]
                agent2_output = _parse_json_output(output_text, "Agent2")
                session.agent2_outputs.append(agent2_output)

                innovations = agent2_output.get("innovations", [])
                for innovation in innovations:
                    if "source" not in innovation:
                        innovation["source"] = _build_paper_label(paper)
                    session.innovations.append(innovation)

            update_progress("Step5: 汇总创新点...")
            update_progress(f"共汇总 {len(session.innovations)} 条创新点")

            update_progress("Step6: 并行执行创新点适配分析...")
            agent3_requests = []
            for idx, innovation in enumerate(session.innovations, 1):
                instructions = load_prompt("paper_search_agent3.yaml")
                context = (
                    f"{instructions}\n\n"
                    f"[innovation detail]\n{innovation}\n\n"
                    f"[user research field description]\n{session.research_text}\n\n"
                )
                request = submit_llm_request(
                    client=client,
                    instructions="Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON.",
                    input_text=context,
                    enable_reasoning=True,
                    temperature=0.4,
                    agent_name=f"PaperSearchAgent3_{idx}",
                )
                agent3_requests.append((innovation, request))

            session.agent3_outputs = []
            session.adaptable = []
            session.not_adaptable = []
            for innovation, request in agent3_requests:
                result, error = request.wait()
                if error or not result:
                    update_progress("Agent3 失败：跳过该创新点")
                    continue
                output_text = result[0]
                agent3_output = _parse_json_output(output_text, "Agent3")
                session.agent3_outputs.append(agent3_output)

                record = {
                    "innovation_detail": innovation.get("innovation_detail") or innovation.get("detail") or str(innovation),
                    "source": innovation.get("source", "Unknown"),
                    "adaptation": agent3_output.get("adaptation", ""),
                    "is_applicable": agent3_output.get("is_applicable", False),
                }
                if record["is_applicable"]:
                    session.adaptable.append(record)
                else:
                    session.not_adaptable.append(record)

            update_progress("Step7: 已完成可移植/不可移植分类")

            update_progress("Step8: 生成组合建议...")
            if session.adaptable:
                instructions = load_prompt("paper_search_agent4.yaml")
                context = (
                    f"{instructions}\n\n"
                    f"[adaptable innovations]\n{json.dumps(session.adaptable, ensure_ascii=False)}\n\n"
                    f"[user research field description]\n{session.research_text}\n\n"
                )
                output_text, _ = client.generate(
                    instructions="Please think very carefully and rigorously before answering, and never fabricate anything. Output ONLY valid JSON.",
                    input_text=context,
                    enable_reasoning=True,
                    temperature=0.4,
                    agent_name="PaperSearchAgent4",
                )
                session.combination_output = _parse_json_output(output_text, "Agent4")
            else:
                session.combination_output = {
                    "best_combination": [],
                    "reason": "未找到可移植的创新点，无法给出组合建议。",
                }

            update_progress("Workflow complete!")
            session.overall_status = ProcessStatus.COMPLETED

            return {
                "adaptable": session.adaptable,
                "not_adaptable": session.not_adaptable,
                "combination": session.combination_output,
            }

        except Exception as e:
            session.overall_status = ProcessStatus.ERROR
            session.progress_message = f"Error: {str(e)}"
            update_progress(f"Workflow failed: {str(e)}")
            raise


paper_search_service = PaperSearchService()
