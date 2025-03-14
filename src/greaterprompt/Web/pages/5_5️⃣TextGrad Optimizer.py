import json

from greaterprompt.optimizer import TextGradOptimizer, GreaterDataloader

import torch
import streamlit as st
from torch.nn import functional as F


CUDA_DEVICES = torch.cuda.device_count()

with st.sidebar:
    settings = st.markdown("<h3>🛠️ Advanced Settings</h3>", unsafe_allow_html=True)
    device = st.selectbox("Device", ["cpu", *[f"cuda:{i}" for i in range(CUDA_DEVICES)]], index=1)

st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🤩 Optimize with TextGradOptimizer</h1>", unsafe_allow_html=True)
evaluation_engine = st.text_input("Evaluation Engine", key="Evaluation Engine", value='meta-llama/Meta-Llama-3-8B-Instruct')
test_engine = st.text_input("Test Engine", key="Test Engine", value='meta-llama/Meta-Llama-3-8B-Instruct')
uploaded_file = st.file_uploader("Upload a jsonl input file", type=("jsonl"))

p_init = st.text_input("P Initial", value="")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("⚡ Start Optimization", type="primary", use_container_width=True)

if not uploaded_file and start_button:
    st.info("Please upload a jsonl file to start")

if p_init.strip() == "" and start_button:
    st.info("Please enter a valid P Initial to start")

if uploaded_file and p_init.strip() and start_button:
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().split('\n')

    inputs = []
    
    for line in lines:
        if line.strip():
            try:
                inputs.append(json.loads(line))
            except json.JSONDecodeError:
                st.error(f"Invalid JSON in line: {line}")

    with st.status("⚡ Optimizing...", expanded=True) as status:
        textgrad_optimizer = TextGradOptimizer({
            "evaluation_engine": evaluation_engine,
            "test_engine": test_engine,
            "device": device
        })
        dataset = GreaterDataloader(custom_inputs=inputs)

        output = textgrad_optimizer.optimize(p_init, dataset)

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
