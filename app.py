import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 未安装 dotenv，跳过

if "--device" in sys.argv:
    try:
        device_idx = sys.argv.index("--device")
        device_value = sys.argv[device_idx + 1]
        os.environ["DOCLING_DEVICE"] = device_value
        print(f"[INFO] DOCLING_DEVICE 已设置为: {device_value}")
    except (IndexError, ValueError):
        print("[WARNING] --device 需要一个值（cpu 或 cuda）")

os.environ["GRADIO_LANGUAGE"] = "zh-CN"

import uuid
import time
import shutil
import threading
import json
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
from fastapi import FastAPI

def _noop(self, app: FastAPI):
    pass

gr.blocks.Blocks._add_health_routes = _noop

from rebuttal_service import (
    rebuttal_service,
    ProcessStatus,
    SessionState,
    QuestionState,
    init_llm_client,
    LogCollector,
)
from paper_reading_service import paper_reading_service


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_CURRENT_DIR, "gradio_uploads")
os.makedirs(SAVE_DIR, exist_ok=True)

def read_gradio_file(file_obj) -> Tuple[Optional[str], Optional[Any]]:
    if file_obj is None:
        return None, None

    if isinstance(file_obj, str):
        return "path", file_obj
    if isinstance(file_obj, dict) and "data" in file_obj:
        return "bytes", file_obj["data"]
    if hasattr(file_obj, "read"):
        return "fileobj", file_obj.read()

    raise ValueError(f"未知的 gr.File 对象格式: {type(file_obj)}")


def save_uploaded_files(pdf_file, review_file, session_id: str) -> Tuple[str, str, str]:
    session_dir = os.path.join(SAVE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    pdf_save_path = os.path.join(session_dir, "paper.pdf")
    review_save_path = os.path.join(session_dir, "review.txt")
    
    pdf_type, pdf_data = read_gradio_file(pdf_file)
    if pdf_type is None:
        raise ValueError("PDF 文件上传失败或格式不正确")
    if pdf_type == "path":
        shutil.copy(pdf_data, pdf_save_path)
    elif pdf_type in ("bytes", "fileobj"):
        with open(pdf_save_path, "wb") as f:
            f.write(pdf_data if isinstance(pdf_data, bytes) else pdf_data)
    
    rev_type, rev_data = read_gradio_file(review_file)
    if rev_type is None:
        raise ValueError("评审文件上传失败或格式不正确")
    review_text = ""
    

    def decode_with_fallback(data: bytes) -> str:
        """尝试多种编码解码字节，优先使用 UTF-8。"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        for enc in encodings:
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        return data.decode('utf-8', errors='replace')
    
    if rev_type == "path":
        with open(rev_data, "rb") as f:
            raw_bytes = f.read()
        review_text = decode_with_fallback(raw_bytes)
    elif rev_type in ("bytes", "fileobj"):
        if isinstance(rev_data, bytes):
            review_text = decode_with_fallback(rev_data)
        else:
            review_text = decode_with_fallback(rev_data)
    
    with open(review_save_path, "w", encoding="utf-8") as f:
        f.write(review_text)
    
    return pdf_save_path, review_save_path, review_text


def save_paper_reading_files(pdf_file, research_field_file, session_id: str) -> Tuple[str, str]:
    """保存论文阅读流程的上传文件"""
    session_dir = os.path.join(SAVE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    pdf_save_path = os.path.join(session_dir, "paper.pdf")
    research_field_save_path = os.path.join(session_dir, "research_field.md")
    
    pdf_type, pdf_data = read_gradio_file(pdf_file)
    if pdf_type is None:
        raise ValueError("PDF 文件上传失败或格式不正确")
    if pdf_type == "path":
        shutil.copy(pdf_data, pdf_save_path)
    elif pdf_type in ("bytes", "fileobj"):
        with open(pdf_save_path, "wb") as f:
            f.write(pdf_data if isinstance(pdf_data, bytes) else pdf_data)
    
    rf_type, rf_data = read_gradio_file(research_field_file)
    if rf_type is None:
        raise ValueError("研究领域文件上传失败或格式不正确")
    
    def decode_with_fallback(data: bytes) -> str:
        """尝试多种编码解码字节，优先使用 UTF-8。"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        for enc in encodings:
            try:
                return data.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return data.decode('utf-8', errors='replace')
    
    research_field_text = ""
    if rf_type == "path":
        with open(rf_data, "rb") as f:
            raw_bytes = f.read()
        research_field_text = decode_with_fallback(raw_bytes)
    elif rf_type in ("bytes", "fileobj"):
        if isinstance(rf_data, bytes):
            research_field_text = decode_with_fallback(rf_data)
        else:
            research_field_text = decode_with_fallback(rf_data)
    
    with open(research_field_save_path, "w", encoding="utf-8") as f:
        f.write(research_field_text)
    
    return pdf_save_path, research_field_save_path


processing_threads: Dict[str, threading.Thread] = {}

# 供应商配置
PROVIDER_CONFIGS = {
    "OpenRouter": {
        "provider_key": "openrouter",
        "env_var": "OPENROUTER_API_KEY",
        "label": "OpenRouter API 密钥",
        "placeholder": "sk-or-v1-...",
    },
    "Qwen (DashScope)": {
        "provider_key": "qwen",
        "env_var": "QWEN_API_KEY",
        "label": "Qwen API 密钥",
        "placeholder": "sk-...",
    },
    "DeepSeek": {
        "provider_key": "deepseek",
        "env_var": "DEEPSEEK_API_KEY",
        "label": "DeepSeek API 密钥",
        "placeholder": "sk-...",
    },
    "OpenAI": {
        "provider_key": "openai",
        "env_var": "OPENAI_API_KEY",
        "label": "OpenAI API 密钥",
        "placeholder": "sk-...",
    },
    "Gemini": {
        "provider_key": "gemini",
        "env_var": "GEMINI_API_KEY",
        "label": "Gemini API 密钥",
        "placeholder": "AIza...",
    },
    "ZhiPu (GLM)": {
        "provider_key": "zhipu",
        "env_var": "ZHIPUAI_API_KEY",
        "label": "智谱 API 密钥",
        "placeholder": "...",
    },
}

# 各供应商模型选项
MODEL_CHOICES_BY_PROVIDER = {
    "OpenRouter": {
        "Gemini 3 Flash": "google/gemini-3-flash-preview",
        "Grok 4.1 Fast": "x-ai/grok-4.1-fast",
        "GPT-5 Mini": "openai/gpt-5-mini",
        "DeepSeek V3.2": "deepseek/deepseek-chat-v3.2",
        "其他模型": "custom",
    },
    "Qwen (DashScope)": {
        "Qwen-Turbo": "qwen-turbo",
        "Qwen-Plus": "qwen-plus",
        "Qwen-Max": "qwen-max",
        "其他模型": "custom",
    },
    "DeepSeek": {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Reasoner": "deepseek-reasoner",
        "其他模型": "custom",
    },
    "OpenAI": {
        "GPT-5.2": "gpt-5.2",
        "GPT-5 Mini": "gpt-5-mini",
        "其他模型": "custom",
    },
    "Gemini": {
        "Gemini-3-Pro": "gemini-3-pro-preview",
        "Gemini-3-Flash": "models/gemini-3-flash-preview",
        "其他模型": "custom",
    },
    "ZhiPu (GLM)": {
        "GLM-4.7": "glm-4.7",
        "其他模型": "custom",
    },
}




def get_api_key_for_provider(provider: str) -> str:
    """从环境变量获取指定供应商的 API 密钥"""
    config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["OpenRouter"])
    return os.environ.get(config["env_var"], "")


def get_default_model_for_provider(provider: str) -> str:
    """获取指定供应商的默认模型"""
    models = MODEL_CHOICES_BY_PROVIDER.get(provider, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
    # 返回第一个模型（排除“其他模型”）
    for name, value in models.items():
        if name != "其他模型":
            return name
    return list(models.keys())[0]


def start_analysis(pdf_file, review_file, provider_choice, api_key, model_choice, custom_model):
    if not pdf_file or not review_file:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ 请上传论文 PDF 和评审文件！",
            gr.Timer(active=False),  
        )
    
    if not api_key or not api_key.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ 请输入 API 密钥！",
            gr.Timer(active=False),
        )
    
    # 从配置获取供应商 key
    provider_config = PROVIDER_CONFIGS.get(provider_choice, PROVIDER_CONFIGS["OpenRouter"])
    provider_key = provider_config["provider_key"]
    
    # 获取该供应商的模型选项
    model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider_choice, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
    
    if model_choice == "其他模型":
        if not custom_model or not custom_model.strip():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                "⚠️ 请输入自定义模型名称！",
                gr.Timer(active=False),
            )
        selected_model = custom_model.strip()
    else:
        selected_model = model_choices.get(model_choice, list(model_choices.values())[0])
    
    session_id = str(uuid.uuid4())[:8]
    
    try:
        init_llm_client(api_key=api_key.strip(), provider=provider_key, model=selected_model)
        pdf_path, review_path, _ = save_uploaded_files(pdf_file, review_file, session_id)
        session = rebuttal_service.create_session(session_id, pdf_path, review_path)
        
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            {"session_id": session_id, "current_idx": 0},
            "📤 文件上传成功，正在初始化分析...",
            gr.Timer(active=True),  
        )
        
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"❌ 处理失败：{str(e)}",
            gr.Timer(active=False),
        )


def run_initial_analysis(session_state):
    if not session_state:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            session_state,
            "❌ 会话状态丢失",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False), 
        )
    
    session_id = session_state.get("session_id")
    
    try:
        session = rebuttal_service.run_initial_analysis(session_id)
        rebuttal_service.process_all_questions_parallel(session_id, max_workers=3)
        session = rebuttal_service.get_session(session_id)
        
        session_state["current_idx"] = 0
        q_state = session.questions[0]
        
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        # 根据是否为最后一个问题决定按钮文案
        is_last_question = len(session.questions) == 1
        btn_text = "📝 生成最终回复" if is_last_question else "✅ 已满足，下一条问题"
        
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            session_state,
            "",
            f"### 问题 1 / {len(session.questions)}",
            q_state.question_text,
            strategy_content,
            strategy_content,
            "",
            f"📝 已修订 {q_state.revision_count} 次",
            gr.update(interactive=True),
            history_text,
            gr.Timer(active=False),
            gr.update(value=btn_text),
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            session_state,
            f"❌ 分析失败：{str(e)}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False),
            gr.update(),
        )


def regenerate_strategy(feedback_text, session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            session_state,
        )
    
    if not feedback_text or not feedback_text.strip():
        return (
            gr.update(),
            gr.update(),
            "⚠️ 请输入反馈",
            gr.update(),
            session_state,
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        q_state = rebuttal_service.revise_with_feedback(
            session_id, 
            current_idx, 
            feedback_text.strip()
        )
        
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        return (
            strategy_content,
            strategy_content,
            "",
            f"📝 已修订 {q_state.revision_count} 次 ✓ 已应用最新修订",
            history_text,
            session_state,
        )
        
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"❌ 修订失败：{str(e)}",
            gr.update(),
            session_state,
        )


def format_feedback_history(history: list) -> str:
    if not history:
        return "*尚无修订*"
    
    lines = []
    for i, record in enumerate(history, 1):
        feedback = record.get("feedback", "")
        if len(feedback) > 100:
            feedback = feedback[:100] + "..."
        lines.append(f"**#{i}** {feedback}")
    
    return "\n\n".join(lines)


def generate_strategy_summary(session) -> str:
    lines = []
    lines.append(" 本文档包含所有问题的回复策略与待办清单\n")
    lines.append("=" * 60 + "\n")
    
    for q in session.questions:
        lines.append(f"## 问题{q.question_id}: {q.question_text[:100]}{'...' if len(q.question_text) > 100 else ''}")
        lines.append("")
        lines.append("### 回复策略与待办清单")
        lines.append("")
        lines.append(q.agent7_output if q.agent7_output else "**尚未生成**")
        lines.append("")
        if q.revision_count > 0:
            lines.append(f"> 📝 已修订 {q.revision_count} 次")
        lines.append("")
        lines.append("-" * 40)
        lines.append("")
    
    return "\n".join(lines)


def skip_question(session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        session = rebuttal_service.get_session(session_id)
        
        if not session:
            raise ValueError(f"未找到会话 {session_id}")
        
        rebuttal_service.mark_question_satisfied(session_id, current_idx)
        
        next_idx = current_idx + 1
        
        if next_idx < len(session.questions):
            q_state = session.questions[next_idx]
            session_state["current_idx"] = next_idx
            
            history_text = format_feedback_history(q_state.feedback_history)
            
            strategy_content = q_state.agent7_output or ""
            
            # 根据是否为最后一个问题决定按钮文案
            is_last_question = (next_idx + 1) == len(session.questions)
            btn_text = "📝 生成最终回复" if is_last_question else "✅ 已满足，下一条问题"
            
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                session_state,
                f"### 问题 {next_idx + 1} / {len(session.questions)}",
                q_state.question_text,
                strategy_content,
                strategy_content,
                "",
                f"📝 已修订 {q_state.revision_count} 次",
                gr.update(interactive=True),
                history_text,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value=btn_text),
            )
        else:
            strategy_summary = generate_strategy_summary(session)
            final_text = rebuttal_service.generate_final_rebuttal(session_id)
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                session_state,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                strategy_summary, strategy_summary, final_text, final_text,
                gr.update(),
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), f"❌ 处理失败：{str(e)}",
            gr.update(),
        )


def confirm_and_next(strategy_text, session_state):
    if not session_state:
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    session_id = session_state.get("session_id")
    current_idx = session_state.get("current_idx", 0)
    
    try:
        session = rebuttal_service.get_session(session_id)
        
        if not session:
            raise ValueError(f"未找到会话 {session_id}")
        
        session.questions[current_idx].agent7_output = strategy_text
        rebuttal_service.mark_question_satisfied(session_id, current_idx)
        
        next_idx = current_idx + 1
        
        if next_idx < len(session.questions):
            q_state = session.questions[next_idx]
            session_state["current_idx"] = next_idx
            
            history_text = format_feedback_history(q_state.feedback_history)
            
            strategy_content = q_state.agent7_output or ""
            
            # 根据是否为最后一个问题决定按钮文案
            is_last_question = (next_idx + 1) == len(session.questions)
            btn_text = "📝 生成最终回复" if is_last_question else "✅ 已满足，下一条问题"
            
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                session_state,
                f"### 问题 {next_idx + 1} / {len(session.questions)}",
                q_state.question_text,
                strategy_content,
                strategy_content,
                "",
                f"📝 已修订 {q_state.revision_count} 次",
                gr.update(interactive=True),
                history_text,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(value=btn_text),
            )
        else:
            strategy_summary = generate_strategy_summary(session)
            final_text = rebuttal_service.generate_final_rebuttal(session_id)
            
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                session_state,
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                strategy_summary, strategy_summary, final_text, final_text,
                gr.update(),
            )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            session_state,
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), f"❌ 处理失败：{str(e)}",
            gr.update(),
        )


def restart_session():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        "",
        None, None,
    )


def get_active_sessions_choices():
    """获取会话下拉选项"""
    sessions = rebuttal_service.list_active_sessions()
    if not sessions:
        return []
    return [(s["display_text"], s["session_id"]) for s in sessions]


def refresh_session_list():
    """刷新会话下拉选项"""
    choices = get_active_sessions_choices()
    if not choices:
        return gr.update(choices=[], value=None), "📭 未发现活动会话"
    return gr.update(choices=choices, value=choices[0][1]), f"🔄 发现 {len(choices)} 个活动会话"


def resume_session(session_id_to_resume, provider_choice, api_key):
    """页面刷新后恢复已有会话"""
    if not session_id_to_resume:
        return (
            gr.update(),  # upload_col
            gr.update(),  # loading_col
            gr.update(),  # interact_col
            gr.update(),  # result_col
            None,         # session_state
            "⚠️ 请选择要恢复的会话！",  # upload_status
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),  # confirm_btn
        )
    
    if not api_key or not api_key.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ 恢复前请输入 API 密钥！",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),
        )
    
    try:
        # 使用提供的凭据初始化 LLM 客户端
        provider_config = PROVIDER_CONFIGS.get(provider_choice, PROVIDER_CONFIGS["OpenRouter"])
        provider_key = provider_config["provider_key"]
        model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider_choice, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
        default_model = list(model_choices.values())[0]
        init_llm_client(api_key=api_key.strip(), provider=provider_key, model=default_model)
        
        session = rebuttal_service.get_session(session_id_to_resume)
        if not session:
            session = rebuttal_service.restore_session_from_disk(session_id_to_resume)
        if not session:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                f"❌ 未找到会话 {session_id_to_resume}！",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # 检查会话是否已有问题被处理
        if not session.questions:
            return (
                gr.update(),
                gr.update(visible=True),  # Show loading page
                gr.update(),
                gr.update(),
                {"session_id": session_id_to_resume, "current_idx": 0},
                "📤 已找到会话但仍在处理中，请稍候...",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # 找到第一个未处理或未满意的问题
        resume_idx = 0
        for i, q in enumerate(session.questions):
            if not q.is_satisfied and q.agent7_output:
                resume_idx = i
                break
            elif q.is_satisfied:
                resume_idx = i + 1
        
        # 如果全部问题已满足，跳转到结果页
        if resume_idx >= len(session.questions):
            strategy_summary = generate_strategy_summary(session)
            final_text = session.final_rebuttal or rebuttal_service.generate_final_rebuttal(session_id_to_resume)
            
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),  # 显示结果页
                {"session_id": session_id_to_resume, "current_idx": resume_idx - 1},
                "",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                gr.update(),
            )
        
        # 恢复到问题审阅页面
        q_state = session.questions[resume_idx]
        history_text = format_feedback_history(q_state.feedback_history)
        strategy_content = q_state.agent7_output or ""
        
        is_last_question = (resume_idx + 1) == len(session.questions)
        btn_text = "📝 生成最终回复" if is_last_question else "✅ 已满足，下一条问题"
        
        return (
            gr.update(visible=False),  # upload_col
            gr.update(visible=False),  # loading_col
            gr.update(visible=True),   # interact_col
            gr.update(visible=False),  # result_col
            {"session_id": session_id_to_resume, "current_idx": resume_idx},  # session_state
            "",  # upload_status
            f"### 问题 {resume_idx + 1} / {len(session.questions)}（已恢复）",  # progress_info
            q_state.question_text,  # question_display
            strategy_content,  # strategy_preview
            strategy_content,  # strategy_edit
            "",  # feedback_input
            f"📝 已修订 {q_state.revision_count} 次",  # revision_info
            gr.update(interactive=True),  # regenerate_btn
            history_text,  # feedback_history_display
            gr.update(value=btn_text),  # confirm_btn
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"❌ 恢复会话失败：{str(e)}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(),
        )


def poll_logs(session_state):
    """轮询加载页的实时日志更新"""
    if not session_state:
        return gr.update(), session_state
    
    session_id = session_state.get("session_id")
    if not session_id:
        return gr.update(), session_state
    
    session = rebuttal_service.get_session(session_id)
    if not session or not session.log_collector:
        return gr.update(), session_state
    
    logs = session.log_collector.get_recent(30)
    if not logs:
        return gr.update(), session_state
    
    prev_logs = session_state.get("_prev_logs", "")
    if logs == prev_logs:
        return gr.update(), session_state
    
    session_state["_prev_logs"] = logs
    return logs, session_state


def poll_pr_logs(pr_session_state):
    """轮询论文阅读流程日志"""
    if not pr_session_state:
        return gr.update(), pr_session_state
    
    session_id = pr_session_state.get("session_id")
    if not session_id:
        return gr.update(), pr_session_state
    
    session = paper_reading_service.get_session(session_id)
    if not session or not session.log_collector:
        return gr.update(), pr_session_state
    
    logs = session.log_collector.get_recent(30)
    if not logs:
        return gr.update(), pr_session_state
    
    prev_logs = pr_session_state.get("_prev_logs", "")
    if logs == prev_logs:
        return gr.update(), pr_session_state
    
    pr_session_state["_prev_logs"] = logs
    return logs, pr_session_state


def start_paper_reading(pdf_file, research_field_file, provider_choice, api_key, model_choice, custom_model):
    """启动论文阅读流程"""
    if not pdf_file or not research_field_file:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ 请上传论文 PDF 和研究领域文件！",
            gr.Timer(active=False),
        )
    
    if not api_key or not api_key.strip():
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            "⚠️ 请输入 API 密钥！",
            gr.Timer(active=False),
        )
    
    provider_config = PROVIDER_CONFIGS.get(provider_choice, PROVIDER_CONFIGS["OpenRouter"])
    provider_key = provider_config["provider_key"]
    
    model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider_choice, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
    
    if model_choice == "其他模型":
        if not custom_model or not custom_model.strip():
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                None,
                "⚠️ 请输入自定义模型名称！",
                gr.Timer(active=False),
            )
        selected_model = custom_model.strip()
    else:
        selected_model = model_choices.get(model_choice, list(model_choices.values())[0])
    
    session_id = str(uuid.uuid4())[:8]
    
    try:
        init_llm_client(api_key=api_key.strip(), provider=provider_key, model=selected_model)
        pdf_path, research_field_path = save_paper_reading_files(pdf_file, research_field_file, session_id)
        session = paper_reading_service.create_session(session_id, pdf_path, research_field_path)
        
        pr_session_state = {
            "session_id": session_id,
            "current_innovation_idx": 0,
            "current_keyword_idx": 0,
            "current_agent3_idx": 0,
            "current_agent4_idx": 0,
            "agent2_data": None,
            "agent3_data": None,
            "agent4_data": None
        }
        
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            pr_session_state,
            "📤 文件上传成功，正在初始化分析...",
            gr.Timer(active=True),
        )
        
    except Exception as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            f"❌ 处理失败：{str(e)}",
            gr.Timer(active=False),
        )


def run_paper_reading_workflow(pr_session_state):
    """执行论文阅读流程并更新界面"""
    if not pr_session_state:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            pr_session_state,
            "❌ 会话状态丢失",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False),
        )
    
    session_id = pr_session_state.get("session_id")
    
    try:
        result = paper_reading_service.run_workflow(session_id)
        
        # 更新会话状态数据
        pr_session_state.update({
            "agent2_data": result["agent2"],
            "agent3_data": result["agent3"],
            "agent4_data": result["agent4"]
        })
        
        # 格式化输出
        agent1_formatted = json.dumps(result["agent1"], indent=2, ensure_ascii=False)
        agent2_summary_text = result["agent2"].get("full_summary", "")
        innovations = result["agent2"].get("innovations", [])
        keywords = result["agent2"].get("keywords", [])
        agent2_innovation_text = innovations[0] if innovations else ""
        agent2_keyword_text = keywords[0] if keywords else ""
        agent3_text = result["agent3"][0] if result["agent3"] else ""
        agent4_text = result["agent4"][0] if result["agent4"] else ""
        agent5_formatted = json.dumps(result["agent5"], indent=2, ensure_ascii=False)
        
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            pr_session_state,
            "",
            agent1_formatted,
            agent2_summary_text,
            agent2_innovation_text,
            f"创新点 1/{len(innovations)}" if innovations else "创新点 0/0",
            agent2_keyword_text,
            f"关键词 1/{len(keywords)}" if keywords else "关键词 0/0",
            agent3_text,
            f"创新点分析 1/{len(result['agent3'])}" if result["agent3"] else "创新点分析 0/0",
            agent4_text,
            f"应用价值分析 1/{len(result['agent4'])}" if result["agent4"] else "应用价值分析 0/0",
            agent5_formatted,
            gr.Timer(active=False),
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            pr_session_state,
            f"❌ 分析失败：{str(e)}",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.Timer(active=False),
        )


def update_innovation_display(pr_session_state, direction):
    """更新 Agent2 创新点展示"""
    if not pr_session_state:
        return pr_session_state, gr.update(), gr.update()
    
    current_idx = pr_session_state.get("current_innovation_idx", 0)
    innovations = pr_session_state.get("agent2_data", {}).get("innovations", [])
    
    if not innovations:
        return pr_session_state, gr.update(), gr.update()
    
    if direction == "next":
        current_idx = min(current_idx + 1, len(innovations) - 1)
    elif direction == "prev":
        current_idx = max(current_idx - 1, 0)
    
    pr_session_state["current_innovation_idx"] = current_idx
    current_text = innovations[current_idx] if current_idx < len(innovations) else ""
    
    return (
        pr_session_state,
        current_text,
        f"创新点 {current_idx + 1}/{len(innovations)}"
    )


def update_keyword_display(pr_session_state, direction):
    """更新 Agent2 关键词展示"""
    if not pr_session_state:
        return pr_session_state, gr.update(), gr.update()
    
    current_idx = pr_session_state.get("current_keyword_idx", 0)
    keywords = pr_session_state.get("agent2_data", {}).get("keywords", [])
    
    if not keywords:
        return pr_session_state, gr.update(), gr.update()
    
    if direction == "next":
        current_idx = min(current_idx + 1, len(keywords) - 1)
    elif direction == "prev":
        current_idx = max(current_idx - 1, 0)
    
    pr_session_state["current_keyword_idx"] = current_idx
    current_text = keywords[current_idx] if current_idx < len(keywords) else ""
    
    return (
        pr_session_state,
        current_text,
        f"关键词 {current_idx + 1}/{len(keywords)}"
    )


def update_agent3_display(pr_session_state, direction):
    """更新 Agent3 展示"""
    if not pr_session_state:
        return pr_session_state, gr.update(), gr.update()
    
    current_idx = pr_session_state.get("current_agent3_idx", 0)
    agent3_data = pr_session_state.get("agent3_data", [])
    
    if not agent3_data:
        return pr_session_state, gr.update(), gr.update()
    
    if direction == "next":
        current_idx = min(current_idx + 1, len(agent3_data) - 1)
    elif direction == "prev":
        current_idx = max(current_idx - 1, 0)
    
    pr_session_state["current_agent3_idx"] = current_idx
    current_text = agent3_data[current_idx] if current_idx < len(agent3_data) else ""
    
    return (
        pr_session_state,
        current_text,
        f"创新点分析 {current_idx + 1}/{len(agent3_data)}"
    )


def update_agent4_display(pr_session_state, direction):
    """更新 Agent4 展示"""
    if not pr_session_state:
        return pr_session_state, gr.update(), gr.update()
    
    current_idx = pr_session_state.get("current_agent4_idx", 0)
    agent4_data = pr_session_state.get("agent4_data", [])
    
    if not agent4_data:
        return pr_session_state, gr.update(), gr.update()
    
    if direction == "next":
        current_idx = min(current_idx + 1, len(agent4_data) - 1)
    elif direction == "prev":
        current_idx = max(current_idx - 1, 0)
    
    pr_session_state["current_agent4_idx"] = current_idx
    current_text = agent4_data[current_idx] if current_idx < len(agent4_data) else ""
    
    return (
        pr_session_state,
        current_text,
        f"应用价值分析 {current_idx + 1}/{len(agent4_data)}"
    )



# 应用 CSS
APP_CSS = """
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');
    
    /* 全局字体：英语使用 Georgia，中文使用 Noto Sans SC */
    * {
        font-family: Georgia, 'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
    }
    .prose, .prose * {
        font-family: Georgia, 'Noto Sans SC', 'PingFang SC', 'Hiragino Sans GB', sans-serif !important;
    }
    /* 代码块保持等宽字体 */
    code, pre, .code, pre *, code * {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }
    .strategy-preview {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        line-height: 1.8;
        max-height: 600px;
        overflow-y: auto;
    }
    .strategy-preview h3 {
        color: #1e40af;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
        margin-top: 20px;
    }
    .strategy-preview h4 {
        color: #7c3aed;
        margin-top: 16px;
    }
    .strategy-preview strong {
        color: #1e293b;
        border-radius: 4px; /* 可选：增加轻微高亮关联 */
    }
    .strategy-preview table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
    }
    .strategy-preview th, .strategy-preview td {
        border: 1px solid #e2e8f0;
        padding: 8px 12px;
        text-align: left;
    }
    .strategy-preview th {
        background: #f1f5f9;
    }
    .strategy-edit textarea {
        font-family: 'Consolas', 'Monaco', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .question-box {
        background: linear-gradient(135deg, #fef3c7, #fef9c3);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
    }
    .feedback-box textarea {
        border: 2px solid #4CAF50;
    }
    #log-display {
        background: #f8fafc;
        color: #334155;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
    }
    /* 下载提示动画 */
    @keyframes pulse-glow {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.7;
            transform: scale(1.02);
        }
    }
    @keyframes arrow-bounce {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(5px); }
    }
    .download-tip {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .download-tip em {
        font-style: normal;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    /* 重要警示信息 - 单层样式 */
    .important-warning {
        background: linear-gradient(135deg, #fef2cd, #fff3cd) !important;
        border: 2px solid #ff9800 !important;
        border-left: 6px solid #ff5722 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin: 16px 0 !important;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.25) !important;
    }
    .important-warning * {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    .important-warning p {
        margin: 4px 0 !important;
        color: #5d4037 !important;
        font-weight: 500 !important;
    }
    /* 明亮的下载按钮 */
    #download-strategy-btn, #download-rebuttal-btn {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 16px 24px !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    #download-strategy-btn:hover, #download-rebuttal-btn:hover {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.6) !important;
        transform: translateY(-2px) !important;
    }
"""

with gr.Blocks(title="AI 论文助手") as demo:
    
    session_state = gr.State(None)
    pr_session_state = gr.State(None)
    
    gr.Markdown(
        """
        # AI 论文助手
        
        面向学术论文的分析与处理流程。
        """
    )
    
    with gr.Tabs() as main_tabs:
        with gr.TabItem("论文阅读"):
            gr.Markdown(
                """
                **论文阅读流程：**
                - **上传** - 上传论文 PDF 与研究领域描述（.md 文件）
                - **分析** - 系统通过 5 个智能体分析论文：
                  1. 提取核心动机与创新点
                  2. 细化并验证分析结果
                  3. 逐条深入分析创新点
                  4. 在研究领域内分析应用价值
                  5. 评估写作质量
                - **查看** - 使用导航控件浏览分析结果
                """
            )
            
            with gr.Column(visible=True) as pr_upload_col:
                gr.Markdown("## 📤 配置并上传文件")
                
                with gr.Group():
                    gr.Markdown("### 🔑 API 配置")
                    
                    pr_provider_choice = gr.Dropdown(
                        label="LLM 供应商",
                        choices=list(PROVIDER_CONFIGS.keys()),
                        value="OpenRouter",
                        info="请选择你的 LLM 供应商",
                    )
                    
                    pr_env_api_key = get_api_key_for_provider("OpenRouter")
                    pr_api_key_input = gr.Textbox(
                        label=PROVIDER_CONFIGS["OpenRouter"]["label"],
                        placeholder=f"请输入 API 密钥（{PROVIDER_CONFIGS['OpenRouter']['placeholder']}）",
                        value=pr_env_api_key,
                        type="password",
                        info="API 密钥不会被存储，仅用于本次会话。" + ("（已从 .env 载入）" if pr_env_api_key else "")
                    )
                    
                    def pr_on_provider_change(provider):
                        config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["OpenRouter"])
                        env_key = get_api_key_for_provider(provider)
                        model_choices = MODEL_CHOICES_BY_PROVIDER.get(provider, MODEL_CHOICES_BY_PROVIDER["OpenRouter"])
                        default_model = get_default_model_for_provider(provider)
                        
                        return (
                            gr.update(
                                label=config["label"],
                                placeholder=f"请输入 API 密钥（{config['placeholder']}）",
                                value=env_key,
                                info="API 密钥不会被存储，仅用于本次会话。" + ("（已从 .env 载入）" if env_key else "")
                            ),
                            gr.update(
                                choices=list(model_choices.keys()),
                                value=default_model,
                            ),
                        )
                
                gr.Markdown("---")
                
                with gr.Group():
                    gr.Markdown("### 🤖 模型选择")
                    with gr.Row():
                        pr_model_choice = gr.Dropdown(
                            label="选择模型",
                            choices=list(MODEL_CHOICES_BY_PROVIDER["OpenRouter"].keys()),
                            value="Gemini 3 Flash",
                            info="选择要使用的 LLM 模型",
                            scale=2,
                        )
                        pr_custom_model_input = gr.Textbox(
                            label="自定义模型名称",
                            placeholder="请输入模型名称",
                            visible=False,
                            scale=3,
                        )
                    
                    def pr_toggle_custom_model(choice):
                        return gr.update(visible=(choice == "其他模型"))
                    
                    pr_model_choice.change(
                        fn=pr_toggle_custom_model,
                        inputs=[pr_model_choice],
                        outputs=[pr_custom_model_input],
                    )
                    
                    pr_provider_choice.change(
                        fn=pr_on_provider_change,
                        inputs=[pr_provider_choice],
                        outputs=[pr_api_key_input, pr_model_choice],
                    )
                
                gr.Markdown("---")
                
                gr.Markdown("### 📄 上传文件")
                with gr.Row():
                    pr_pdf_input = gr.File(
                        label="📄 论文 PDF",
                        file_types=[".pdf"],
                        file_count="single",
                    )
                    pr_research_field_input = gr.File(
                        label="📝 研究领域描述（.md）",
                        file_types=[".md"],
                        file_count="single",
                    )
                
                pr_upload_status = gr.Markdown("")
                
                pr_start_btn = gr.Button(
                    "🚀 提交并开始分析",
                    variant="primary",
                    size="lg",
                )
            
            with gr.Column(visible=False) as pr_loading_col:
                gr.Markdown("## ⏳ 正在分析...")
                pr_loading_status = gr.Markdown("初始化中...")
                
                gr.Markdown(
                    """
                    > 📊 **分析流程：**
                    > 1. 将 PDF 转换为 Markdown
                    > 2. Agent1：提取核心动机与创新点
                    > 3. Agent2：细化并验证分析
                    > 4. Agent3：逐条深入分析创新点（并行）
                    > 5. Agent4：分析在研究领域的应用价值（并行）
                    > 6. Agent5：评估写作质量
                    > 7. 输出结果供你查看
                    
                    这可能需要几分钟，请耐心等待...
                    """
                )
                
                gr.Markdown("### 📋 实时日志")
                pr_log_display = gr.Textbox(
                    value="等待开始...",
                    label="",
                    lines=10,
                    max_lines=15,
                    interactive=False,
                    elem_id="log-display",
                )

                pr_log_timer = gr.Timer(value=1.5, active=False)
            
            with gr.Column(visible=False) as pr_result_col:
                gr.Markdown("## 📊 分析结果")
                
                gr.Markdown("### Agent1：核心摘要")
                pr_agent1_output = gr.Textbox(
                    label="Agent1 输出（JSON）",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                )
                
                gr.Markdown("---")
                gr.Markdown("### Agent2：完整摘要与创新点")
                pr_agent2_summary = gr.Textbox(
                    label="完整摘要",
                    lines=6,
                    max_lines=10,
                    interactive=False,
                )
                
                with gr.Row():
                    pr_innovation_prev_btn = gr.Button("◀ 上一条", size="sm")
                    pr_innovation_index = gr.Markdown("创新点 1/1")
                    pr_innovation_next_btn = gr.Button("下一条 ▶", size="sm")
                pr_agent2_innovation = gr.Textbox(
                    label="当前创新点",
                    lines=4,
                    max_lines=8,
                    interactive=False,
                )
                
                with gr.Row():
                    pr_keyword_prev_btn = gr.Button("◀ 上一条", size="sm")
                    pr_keyword_index = gr.Markdown("关键词 1/1")
                    pr_keyword_next_btn = gr.Button("下一条 ▶", size="sm")
                pr_agent2_keyword = gr.Textbox(
                    label="当前关键词",
                    lines=2,
                    max_lines=4,
                    interactive=False,
                )
                
                gr.Markdown("---")
                gr.Markdown("### Agent3：创新点分析")
                with gr.Row():
                    pr_agent3_prev_btn = gr.Button("◀ 上一条", size="sm")
                    pr_agent3_index = gr.Markdown("创新点分析 1/1")
                    pr_agent3_next_btn = gr.Button("下一条 ▶", size="sm")
                pr_agent3_output = gr.Textbox(
                    label="当前创新点分析",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )
                
                gr.Markdown("---")
                gr.Markdown("### Agent4：应用价值分析")
                with gr.Row():
                    pr_agent4_prev_btn = gr.Button("◀ 上一条", size="sm")
                    pr_agent4_index = gr.Markdown("应用价值分析 1/1")
                    pr_agent4_next_btn = gr.Button("下一条 ▶", size="sm")
                pr_agent4_output = gr.Textbox(
                    label="当前应用价值分析",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )
                
                gr.Markdown("---")
                gr.Markdown("### Agent5：写作质量评估")
                pr_agent5_output = gr.Textbox(
                    label="Agent5 输出（JSON）",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                )
            
            pr_start_btn.click(
                fn=start_paper_reading,
                inputs=[pr_pdf_input, pr_research_field_input, pr_provider_choice, pr_api_key_input, pr_model_choice, pr_custom_model_input],
                outputs=[
                    pr_upload_col, pr_loading_col, pr_result_col,
                    pr_session_state, pr_upload_status, pr_log_timer,
                ],
            ).then(
                fn=run_paper_reading_workflow,
                inputs=[pr_session_state],
                outputs=[
                    pr_upload_col, pr_loading_col, pr_result_col,
                    pr_session_state, pr_loading_status,
                    pr_agent1_output,
                    pr_agent2_summary,
                    pr_agent2_innovation,
                    pr_innovation_index,
                    pr_agent2_keyword,
                    pr_keyword_index,
                    pr_agent3_output,
                    pr_agent3_index,
                    pr_agent4_output,
                    pr_agent4_index,
                    pr_agent5_output,
                    pr_log_timer,
                ],
            )
            
            pr_log_timer.tick(
                fn=poll_pr_logs,
                inputs=[pr_session_state],
                outputs=[pr_log_display, pr_session_state],
            )
            
            pr_innovation_prev_btn.click(
                fn=lambda state: update_innovation_display(state, "prev"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent2_innovation, pr_innovation_index],
            )
            
            pr_innovation_next_btn.click(
                fn=lambda state: update_innovation_display(state, "next"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent2_innovation, pr_innovation_index],
            )
            
            pr_keyword_prev_btn.click(
                fn=lambda state: update_keyword_display(state, "prev"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent2_keyword, pr_keyword_index],
            )
            
            pr_keyword_next_btn.click(
                fn=lambda state: update_keyword_display(state, "next"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent2_keyword, pr_keyword_index],
            )
            
            pr_agent3_prev_btn.click(
                fn=lambda state: update_agent3_display(state, "prev"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent3_output, pr_agent3_index],
            )
            
            pr_agent3_next_btn.click(
                fn=lambda state: update_agent3_display(state, "next"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent3_output, pr_agent3_index],
            )
            
            pr_agent4_prev_btn.click(
                fn=lambda state: update_agent4_display(state, "prev"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent4_output, pr_agent4_index],
            )
            
            pr_agent4_next_btn.click(
                fn=lambda state: update_agent4_display(state, "next"),
                inputs=[pr_session_state],
                outputs=[pr_session_state, pr_agent4_output, pr_agent4_index],
            )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 论文助手")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="docling PDF 处理设备（cpu 或 cuda）")
    
    args = parser.parse_args()
    
    device_used = os.environ.get("DOCLING_DEVICE", "cpu")
    
    print(f"\n🚀 启动 AI 论文助手")
    print(f"   地址: http://localhost:{args.port}")
    print(f"   设备: {device_used.upper()}")
    print(f"   共享: {'是' if args.share else '否'}\n")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),  # Moved here for Gradio 6.0
        css=APP_CSS,             # Moved here for Gradio 6.0
    )
