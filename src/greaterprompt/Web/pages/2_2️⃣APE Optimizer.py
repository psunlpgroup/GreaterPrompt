import json

from greaterprompt.optimizer import ApeOptimizer, GreaterDataloader

import openai
import streamlit as st


with st.sidebar:
    openai_key = st.markdown("<h3>🔐 OpenAI Key</h3>", unsafe_allow_html=True)
    openai_key = st.text_input("OpenAI Key", type="password")
    settings = st.markdown("<h3>🛠️ Advanced Settings</h3>", unsafe_allow_html=True)
    task_model = st.selectbox("Task Model", ["openai_gpt35_turbo_instruct", "openai_gpt4", "openai_gpt4_turbo", "openai_gpt4o", "openai_gpt4o_mini"], index=0)
    optim_model = st.selectbox("Optim Model", ["openai_gpt35_turbo_instruct", "openai_gpt4", "openai_gpt4_turbo", "openai_gpt4o", "openai_gpt4o_mini"], index=2)

st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🤩 Optimize with APE Optimizer</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a jsonl input file", type=("jsonl"))
p_init = st.text_input("P Initial")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("⚡ Start Optimization", type="primary", use_container_width=True)

if not openai_key and start_button:
    st.info("Please enter an OpenAI key to start")

if not uploaded_file and start_button:
    st.info("Please upload a jsonl file to start")

if p_init.strip() == "" and start_button:
    st.info("Please enter a valid P Initial to start")

if openai_key and uploaded_file and p_init.strip() and start_button:
    openai.api_key = openai_key
    
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().split('\n')

    inputs = []
    
    for line in lines:
        if line.strip():
            try:
                inputs.append(json.loads(line))
            except json.JSONDecodeError:
                st.error(f"Invalid JSON in line: {line}")

    optimize_config = {
        "task_model": task_model,
        "optim_model": optim_model,
    }

    with st.status("⚡ Optimizing...", expanded=True) as status:
        ape_optimizer = ApeOptimizer(optimize_config=optimize_config)
        dataset = GreaterDataloader(custom_inputs=inputs)

        output = ape_optimizer.optimize(dataset, p_init)

        status.update(label="😄 Optimization complete!", state="complete")
        st.markdown("<h5 style='text-align: left; white-space: nowrap;'>🎯 Optimization Result:</h5>", unsafe_allow_html=True)
        st.markdown(f"<p>{output}</p>", unsafe_allow_html=True)

        if output:
            st.download_button(
                label="📥 Download Optimized Result",
                data=json.dumps(output, indent=2, ensure_ascii=False),
                file_name="optimized_prompt.json",
                mime="application/json"
            )
