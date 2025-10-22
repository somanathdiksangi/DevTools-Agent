# app.py
import streamlit as st
import json
import pandas as pd
from main import (
    run_self_heal, get_traces, evaluator, evaluator_aggregate, preflight,
    cloud_cli, open_pr_and_ci, autogen_template_loader
)

st.set_page_config(page_title="Self-Healing DevTools Agent", layout="wide")
st.title("Self-Healing DevTools Agent with Tool Router")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Run Demo", "Telemetry", "ToolRouter Dashboard", "Evaluator", "Cloud CLI"]
)

with tab1:
    st.subheader("Execute and Self-Heal")
    code = st.text_area("Paste a Python function", value="""
def divide_two_numbers(a, b):
    return a/b
""", height=180)
    args = st.text_input("Arguments (JSON list)", value='[10,0]')
    if st.button("Run self-heal"):
        ns = {}
        try:
            exec(code, ns)
            fn_name = [k for k in ns.keys() if callable(ns.get(k))][-1]
            fn = ns[fn_name]
            res = run_self_heal(fn, eval(args), function_string_override=code)
            st.success("Run complete")
            st.json(res)
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("Normalized Traces")
    traces = get_traces()
    st.caption("Each span has name, type, start/end timestamps, duration, status, and context/meta.")
    st.json(traces)
    tool_events = [e for e in traces if e.get("type") == "tool"]
    if tool_events:
        df = pd.DataFrame([{
            "app": e["ctx"].get("app"),
            "action": e["ctx"].get("action"),
            "status": e["status"],
            "latency_ms": e.get("meta", {}).get("latency_ms", 0.0)
        } for e in tool_events])
        st.bar_chart(df, x="action", y="latency_ms")

with tab3:
    st.subheader("ToolRouter Dashboard")
    traces = get_traces()
    tool_events = [e for e in traces if e.get("type") == "tool"]
    if tool_events:
        df = pd.DataFrame([{
            "app": e["ctx"].get("app"),
            "action": e["ctx"].get("action"),
            "status": e["status"],
            "latency_ms": e.get("meta", {}).get("latency_ms", 0.0),
            "params_hash": e["ctx"].get("params_hash"),
        } for e in tool_events])
        apps = ["All"] + sorted(df["app"].dropna().unique().tolist())
        actions = ["All"] + sorted(df["action"].dropna().unique().tolist())
        app_filter = st.selectbox("App", apps, index=0)
        action_filter = st.selectbox("Action", actions, index=0)
        view = df.copy()
        if app_filter != "All":
            view = view[view["app"] == app_filter]
        if action_filter != "All":
            view = view[view["action"] == action_filter]
        st.dataframe(view, use_container_width=True)
    else:
        st.info("No ToolRouter calls recorded yet.")

with tab4:
    st.subheader("Preflight & Evaluator")
    if st.button("Run preflight"):
        st.json(preflight())
    colA, colB = st.columns(2)
    with colA:
        seed = st.number_input("Seed", value=7, step=1)
        if st.button("Run evaluator (single seed)"):
            st.json(evaluator(seed=int(seed)))
    with colB:
        if st.button("Run aggregate (7,13,21)"):
            st.json(evaluator_aggregate())
    st.caption("Scorecards and preflight outputs are written to artifacts/ for CI.")

with tab5:
    st.subheader("Cloud Storage via Tool Router")
    action = st.selectbox("Action", ["list", "sync", "presign", "version"])
    params = st.text_area("Params (JSON)", value="{}")
    confirm = True
    if action in ("sync",):
        confirm = st.checkbox("Confirm write action", value=False)
    if st.button("Execute"):
        if action in ("sync",) and not confirm:
            st.warning("Please confirm to run write actions.")
        else:
            try:
                out = cloud_cli(action, **json.loads(params))
                st.json(out)
            except Exception as e:
                st.error(str(e))

st.divider()
st.subheader("Autogen Template Loader (YAML)")
yaml_text = st.text_area("Template YAML", value="""
agents:
  - name: fixer
    role: writes patches
    tools: [git_pr, ci_trigger]
tools:
  - name: git_pr
    app: github
    action: open_pr
  - name: ci_trigger
    app: circleci
    action: trigger_pipeline
plan: run fixer -> open PR -> trigger CI
""", height=180)
if st.button("Load template"):
    st.json(autogen_template_loader(yaml_text))
