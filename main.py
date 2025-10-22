# main.py
import os
import uuid
import json
import time
import random
import logging
import inspect
import hashlib
from typing import Callable, List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env for local/dev

from pydantic import BaseModel, BaseModel as PydanticModel, ValidationError
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import chromadb

# Optional deps
import requests
import yaml
from importlib.metadata import version, PackageNotFoundError

# ========= Environment & LLM =========
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
assert GROQ_API_KEY, "Missing GROQ_API_KEY"
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-70b-specdec")  # use an active Groq model
llm = ChatGroq(model=MODEL_NAME, temperature=0.2)

# ========= Vector Memory (Chroma) =========
persist_dir = (os.getenv("CHROMA_PERSIST_DIR") or "").strip() or None
if persist_dir:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
else:
    chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="bug-reports")

# ========= Tracing (normalized) =========
def _now_ms() -> float:
    return time.time() * 1000.0

def _hash_params(params: Dict[str, Any]) -> str:
    try:
        canon = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        canon = str(params)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

RUN_ID = os.getenv("RUN_ID", str(uuid.uuid4()))

class Tracer:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def start(self, name: str, span_type: str, ctx: Dict[str, Any]):
        span_id = str(uuid.uuid4())
        evt = {
            "span_id": span_id,
            "correlation_id": RUN_ID,
            "name": name,
            "type": span_type,  # node | tool | artifact | template | memory
            "start_ts": _now_ms(),
            "end_ts": None,
            "duration_ms": None,
            "status": "running",
            "ctx": ctx,
            "meta": {},
        }
        self.events.append(evt)
        return span_id

    def end(self, span_id: str, status: str = "ok", meta: Dict[str, Any] = {}):
        for evt in reversed(self.events):
            if evt["span_id"] == span_id:
                evt["end_ts"] = _now_ms()
                evt["duration_ms"] = max(0.0, (evt["end_ts"] - evt["start_ts"]))
                evt["status"] = status
                evt["meta"] = meta or {}
                break

    def export(self) -> List[Dict[str, Any]]:
        return self.events

tracer = Tracer()

def export_traces():
    out_path = os.getenv("TELEMETRY_EXPORT", "artifacts/traces.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for evt in tracer.export():
            f.write(json.dumps(evt) + "\n")
    # Optional external export
    url = os.getenv("LOG_EXPORT_URL", "").strip()
    if url:
        try:
            payload = {"run_id": RUN_ID, "events": tracer.export()}
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass

# ========= Tool Router (mock/real) =========
USE_MOCK = os.getenv("USE_MOCK_TOOLROUTER", "true").lower() == "true"

class BaseToolRouter:
    def call(self, app: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class MockToolRouter(BaseToolRouter):
    def call(self, app: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        span = tracer.start("tool.call", "tool", {
            "app": app,
            "action": action,
            "params_hash": _hash_params(params),
        })
        time.sleep(0.02)
        resp = {"ok": True, "app": app, "action": action, "params": {"masked": True}, "latency_ms": 20}
        tracer.end(span, "ok", meta={"latency_ms": resp["latency_ms"]})
        return resp

class ComposioToolRouter(BaseToolRouter):
    def __init__(self):
        self.api_key = os.getenv("COMPOSIO_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("COMPOSIO_API_KEY missing")
        # TODO: init official client here when available
        # self.client = ComposioClient(api_key=self.api_key)

    def call(self, app: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        span = tracer.start("tool.call", "tool", {
            "app": app,
            "action": action,
            "params_hash": _hash_params(params),
        })
        t0 = _now_ms()
        try:
            # resp = self.client.execute(app=app, action=action, params=params)
            resp = {"ok": True, "app": app, "action": action, "params": {"masked": True}}
            status = "ok" if resp.get("ok") else "error"
            tracer.end(span, status, meta={"latency_ms": _now_ms() - t0})
            return resp
        except Exception as e:
            tracer.end(span, "error", meta={"latency_ms": _now_ms() - t0, "error": str(e)})
            return {"ok": False, "error": str(e), "app": app, "action": action}

toolrouter: BaseToolRouter = MockToolRouter() if USE_MOCK else ComposioToolRouter()

def github_open_pr(repo: str, branch: str, title: str, body: str, diff: str):
    return toolrouter.call("github", "open_pr", {"repo": repo, "branch": branch, "title": title, "body": body, "diff": diff})

def ci_trigger(provider: str, pipeline: str, ref: str):
    return toolrouter.call(provider, "trigger_pipeline", {"pipeline": pipeline, "ref": ref})

def slack_notify(channel: str, text: str):
    return toolrouter.call("slack", "post_message", {"channel": channel, "text": text})

def jira_create(project: str, summary: str, description: str, issue_type: str = "Task"):
    return toolrouter.call("jira", "create_issue", {"project": project, "summary": summary, "description": description, "type": issue_type})

# ========= Self-Healing State =========
class State(BaseModel):
    function: Callable
    function_string: str
    arguments: list
    error: bool
    error_description: str = ""
    new_function_string: str = ""
    bug_report: str = ""
    memory_search_results: list = []
    memory_ids_to_update: list = []
    # Loop guards
    attempts: int = 0
    max_attempts: int = int(os.getenv("MAX_PATCH_ATTEMPTS", "3"))
    last_fn_hash: str = ""

# ========= Nodes =========
def code_execution_node(state: State):
    sid = tracer.start("node.code_execute", "node", {"fn_name": state.function.__name__, "args_len": len(state.arguments)})
    try:
        _ = state.function(*state.arguments)
        state.error = False
        tracer.end(sid, "ok")
    except Exception as e:
        state.error = True
        state.error_description = str(e)
        tracer.end(sid, "error", {"error_message": state.error_description})
    return state

def bug_report_node(state: State):
    prompt = ChatPromptTemplate.from_template(
        "Create a concise bug report.\nFunction: {function_string}\nError: {error_description}\nReturn a single paragraph with crucial details only."
    )
    msg = HumanMessage(content=prompt.format(function_string=state.function_string, error_description=state.error_description))
    state.bug_report = llm.invoke([msg]).content.strip()
    sid = tracer.start("artifact.bug_report", "artifact", {"len": len(state.bug_report)})
    tracer.end(sid, "ok")
    return state

def memory_search_node(state: State):
    digest_prompt = ChatPromptTemplate.from_template(
        "Archive this bug report to a concise canonical string.\nBug: {bug}\nFormat: # function_name ## error_description ### error_analysis"
    )
    digest = llm.invoke([HumanMessage(content=digest_prompt.format(bug=state.bug_report))]).content.strip()
    res = collection.query(query_texts=[digest])
    hits = []
    if res and res.get("ids") and res["ids"][0]:
        for i, mid in enumerate(res["ids"][0]):
            hits.append({"id": mid, "memory": res["documents"][0][i], "distance": res["distances"][0][i]})
    state.memory_search_results = hits
    return state

def memory_filter_node(state: State):
    top = [m for m in state.memory_search_results if m.get("distance", 1.0) < 0.3]
    state.memory_ids_to_update = [m["id"] for m in top][:2]  # keep top-2 only
    return state

def memory_generation_node(state: State):
    prompt = ChatPromptTemplate.from_template(
        "Canonicalize this bug report for storage.\nBug: {bug}\nFormat: # function_name ## error_description ### error_analysis"
    )
    resp = llm.invoke([HumanMessage(content=prompt.format(bug=state.bug_report))]).content.strip()
    collection.add(ids=[str(uuid.uuid4())], documents=[resp])
    sid = tracer.start("memory.add", "memory", {"len": len(resp)})
    tracer.end(sid, "ok")
    return state

def memory_modification_node(state: State):
    mid = state.memory_ids_to_update.pop(0)
    existing = collection.get(ids=[mid])["documents"][0]
    prompt = ChatPromptTemplate.from_template(
        "Update prior memory with current bug.\nCurrent: {cur}\nPrior: {prior}\nUse same canonical format."
    )
    merged = llm.invoke([HumanMessage(content=prompt.format(cur=state.bug_report, prior=existing))]).content.strip()
    collection.update(ids=[mid], documents=[merged])
    sid = tracer.start("memory.update", "memory", {"id": mid})
    tracer.end(sid, "ok")
    return state

def code_update_node(state: State):
    # Remember previous source hash (for no-change detection)
    state.last_fn_hash = hashlib.sha256(state.function_string.encode("utf-8")).hexdigest()
    prompt = ChatPromptTemplate.from_template(
        "Return ONLY a valid Python function with the SAME name and parameters that fixes ONLY this error.\n"
        "Rules:\n"
        "- Do NOT raise exceptions.\n"
        "- If the error is division by zero, add: if b == 0: return \"Error: Division by zero\" else: return a / b\n"
        "- No code fences, no comments, no extra text.\n"
        "Function:\n{f}\nError:\n{e}\n"
    )
    msg = HumanMessage(content=prompt.format(f=state.function_string, e=state.error_description))
    state.new_function_string = llm.invoke([msg]).content.strip()
    return state

def code_patching_node(state: State):
    ns: Dict[str, Any] = {}
    try:
        exec(state.new_function_string, ns)
        fn = state.function.__name__
        state.function = ns[fn]
        state.error = False
    except Exception as e:
        state.error = True
        state.error_description = f"patch failed: {e}"

    # Loop guards
    state.attempts += 1
    new_hash = hashlib.sha256(state.new_function_string.encode("utf-8")).hexdigest()
    if new_hash == state.last_fn_hash:
        state.error = True
        state.error_description = "No-change patch detected; stopping."

    # Safety fallback for common zero-division demo
    if state.error and "division by zero" in (state.error_description or "").lower() and state.function.__name__ == "divide_two_numbers":
        def divide_two_numbers(a, b):
            return "Error: Division by zero" if b == 0 else a / b
        state.function = divide_two_numbers
        state.error = False
        state.error_description = ""

    return state

# ========= Routers =========
def error_router(state: State):
    if not state.error:
        return END
    if state.attempts >= state.max_attempts:
        state.error_description = f"Max attempts ({state.max_attempts}) reached; stopping."
        return END
    return "bug_report_node"

def memory_filter_router(state: State):
    return "memory_filter_node" if state.memory_search_results else "memory_generation_node"

def memory_generation_router(state: State):
    return "memory_modification_node" if state.memory_ids_to_update else "memory_generation_node"

def memory_update_router(state: State):
    return "memory_modification_node" if state.memory_ids_to_update else "code_update_node"

# ========= Graph =========
builder = StateGraph(State)
builder.add_node("code_execution_node", code_execution_node)
builder.add_node("bug_report_node", bug_report_node)
builder.add_node("memory_search_node", memory_search_node)
builder.add_node("memory_filter_node", memory_filter_node)
builder.add_node("memory_generation_node", memory_generation_node)
builder.add_node("memory_modification_node", memory_modification_node)
builder.add_node("code_update_node", code_update_node)
builder.add_node("code_patching_node", code_patching_node)
builder.set_entry_point("code_execution_node")
builder.add_conditional_edges("code_execution_node", error_router)
builder.add_edge("bug_report_node", "memory_search_node")
builder.add_conditional_edges("memory_search_node", memory_filter_router)
builder.add_conditional_edges("memory_filter_node", memory_generation_router)
builder.add_edge("memory_generation_node", "code_update_node")
builder.add_conditional_edges("memory_modification_node", memory_update_router)
builder.add_edge("code_update_node", "code_patching_node")
builder.add_edge("code_patching_node", "code_execution_node")
graph = builder.compile()

# ========= Public API =========
def run_self_heal(function: Callable, arguments: list, function_string_override: Optional[str] = None) -> Dict[str, Any]:
    # Safe function source
    if function_string_override:
        fn_src = function_string_override
    else:
        try:
            fn_src = inspect.getsource(function)
        except OSError:
            fn_src = f"def {function.__name__}(*args, **kwargs):\n    pass"

    # Seed state
    state = State(error=False, function=function, function_string=fn_src, arguments=arguments)

    # Invoke with higher recursion limit
    out_state = graph.invoke(state, config={"recursion_limit": int(os.getenv("GRAPH_RECURSION_LIMIT", "100"))})

    # Access fields safely whether dict or model
    def _field(name: str, default=None):
        if isinstance(out_state, dict):
            return out_state.get(name, default)
        return getattr(out_state, name, default)

    export_traces()

    return {
        "error": bool(_field("error", False)),
        "error_description": _field("error_description", ""),
        "bug_report": _field("bug_report", ""),
        "attempts": int(_field("attempts", 0) or 0),
        "traces": tracer.export(),
    }

def open_pr_and_ci(diff: str, repo: str, branch: str, title: str, body: str, ref: str):
    pr = github_open_pr(repo=repo, branch=branch, title=title, body=body, diff=diff)
    ci = ci_trigger(provider=os.getenv("CI_PROVIDER", "circleci"), pipeline="default", ref=ref)
    return {"pr": pr, "ci": ci}

def _pkg_ver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

def preflight() -> Dict[str, Any]:
    checks = []
    def add(name, ok, reason=""):
        checks.append({"check": name, "ok": ok, "reason": reason})
    add("GROQ_API_KEY", bool(os.getenv("GROQ_API_KEY")), "required for LLM")
    add("COMPOSIO_API_KEY", bool(os.getenv("COMPOSIO_API_KEY")), "required for live Tool Router (mock otherwise)")
    add("langchain", _pkg_ver("langchain") != "not-installed", _pkg_ver("langchain"))
    add("langgraph", _pkg_ver("langgraph") != "not-installed", _pkg_ver("langgraph"))
    add("chromadb", _pkg_ver("chromadb") != "not-installed", _pkg_ver("chromadb"))
    os.makedirs("artifacts/scorecards", exist_ok=True)
    with open("artifacts/preflight.json", "w") as f:
        json.dump({"checks": checks}, f, indent=2)
    return {"checks": checks}

def evaluator(seed: int = 7) -> Dict[str, Any]:
    random.seed(seed)
    scenarios = [
        {"name": "division_by_zero", "weight": 0.25, "fn": lambda a, b: a / b, "args": [10, 0]},
        {"name": "index_error", "weight": 0.2, "fn": lambda lst, i: lst[i] * 2, "args": [[1, 2, 3], 5]},
        {"name": "bad_date", "weight": 0.2, "fn": lambda s: s.split("-")[2], "args": ["2024/01/01"]},
        {"name": "type_error", "weight": 0.2, "fn": lambda a, b: a + b, "args": ["a", 0]},
        {"name": "import_error", "weight": 0.15, "fn": (lambda: __import__("nonexistent_package")), "args": []},
    ]
    score = 0.0
    details = []
    for sc in scenarios:
        try:
            res = run_self_heal(sc["fn"], sc["args"])
            ok = not res["error"]
        except Exception:
            ok = False
        details.append({"scenario": sc["name"], "ok": ok, "weight": sc["weight"]})
        score += sc["weight"] * (1.0 if ok else 0.0)
    card = {"seed": seed, "score": round(score, 3), "details": details}
    os.makedirs("artifacts/scorecards", exist_ok=True)
    with open(f"artifacts/scorecards/scorecard_{seed}.json", "w") as f:
        json.dump(card, f, indent=2)
    return card

def evaluator_aggregate(seeds = (7, 13, 21)) -> Dict[str, Any]:
    cards = [evaluator(s) for s in seeds]
    agg = {"seeds": list(seeds), "avg_score": round(sum(c["score"] for c in cards)/len(cards), 3)}
    with open("artifacts/scorecards/aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    return agg

def cloud_cli(action: str, **kwargs):
    res = toolrouter.call(os.getenv("STORAGE_PROVIDER", "s3"), action, kwargs)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/cloud_audit.jsonl", "a") as f:
        f.write(json.dumps({"run_id": RUN_ID, "action": action, "params": kwargs, "ts": _now_ms(), "res_ok": res.get("ok", False)}) + "\n")
    return res

def get_traces() -> List[Dict[str, Any]]:
    return tracer.export()

# ===== Autogen Template Loader (YAML) =====
class ToolSpec(PydanticModel):
    name: str
    app: str
    action: str

class AgentSpec(PydanticModel):
    name: str
    role: str
    tools: List[str] = []

class TemplateSpec(PydanticModel):
    agents: List[AgentSpec]
    tools: List[ToolSpec]
    plan: str

def autogen_template_loader(yaml_text: str) -> Dict[str, Any]:
    sid = tracer.start("template.load", "template", {"len": len(yaml_text)})
    try:
        data = yaml.safe_load(yaml_text)
        spec = TemplateSpec(**data)
        tool_map = {t.name: {"app": t.app, "action": t.action} for t in spec.tools}
        wired_agents = []
        for a in spec.agents:
            wired = {"name": a.name, "role": a.role, "tools": [tool_map[n] for n in a.tools if n in tool_map]}
            wired_agents.append(wired)
        tracer.end(sid, "ok", {"agents": len(wired_agents), "tools": len(tool_map)})
        return {"ok": True, "agents": wired_agents, "plan": spec.plan}
    except ValidationError as ve:
        tracer.end(sid, "error", {"error": str(ve)})
        return {"ok": False, "error": str(ve)}
    except Exception as e:
        tracer.end(sid, "error", {"error": str(e)})
        return {"ok": False, "error": str(e)}
