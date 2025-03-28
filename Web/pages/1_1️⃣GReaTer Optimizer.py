import json
from greaterprompt.optimizer import GreaterOptimizer, GreaterDataloader

import torch
import streamlit as st
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


CUDA_DEVICES = torch.cuda.device_count()

with st.sidebar:
    settings = st.markdown("<h3>🛠️ Advanced Settings</h3>", unsafe_allow_html=True)
    device = st.selectbox("Device", ["cpu", *[f"cuda:{i}" for i in range(CUDA_DEVICES)]], index=1)
    intersect_q = st.number_input("Intersect Q", value=5)
    candidates_topk = st.number_input("Candidates Topk", value=10)
    perplexity_loss = st.selectbox("Perplexity Loss", [True, False])
    perplexity_lambda = st.number_input("Perplexity Lambda", value=0.2)
    filter = st.selectbox("Filter", [True, False])
    rounds = st.number_input("Rounds", value=80)

st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🤩 Optimize with GreaterOptimizer</h1>", unsafe_allow_html=True)
model_path = st.text_input("Model Path", key="model_path", value='google/gemma-2-9b-it')
uploaded_file = st.file_uploader("Upload a jsonl input file", type=("jsonl"))

p_extraction = st.text_input("P Extractor", value="\nNext, only give the exact answer, no extract words or any punctuation:")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("⚡ Start Optimization", type="primary", use_container_width=True)

optimize_config = {
    "intersect_q": intersect_q,
    "candidates_topk": candidates_topk,
    "loss_function": F.cross_entropy,
    "perplexity_loss": perplexity_loss,
    "perplexity_lambda": perplexity_lambda,
    "filter": filter,
    "generate_config": {
        "do_sample": True,
        "temperature": 0.2,
        "max_new_tokens": 512
    }
}

if not uploaded_file and start_button:
    st.info("Please upload a jsonl file to start")

if p_extraction.strip() == "" and start_button:
    st.info("Please enter a valid P Extractor to start")

if uploaded_file and p_extraction.strip() and start_button:
    content = uploaded_file.getvalue().decode("utf-8")
    lines = content.strip().split('\n')

    inputs = []
    
    for line in lines:
        if line.strip():
            try:
                inputs.append(json.loads(line))
            except json.JSONDecodeError:
                st.error(f"Invalid JSON in line: {line}")

    with st.status(f"Loading model from {model_path}...", expanded=True) as status:
        st.write(f"Loading model to {device}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

        st.write("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        
        st.write("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        st.write("Loading optimizer...")
        optimizer = GreaterOptimizer(model=model, tokenizer=tokenizer, optimize_config=optimize_config)

        st.write("Building data inputs...")
        dataset = GreaterDataloader(custom_inputs=inputs)
        
        status.update(label="🥰 Model and Optimizer loaded successfully! Everything is ready now, start optimizing!", state="complete")

    with st.status("⚡ Optimizing...", expanded=True) as status:
        progress_bar = st.progress(0)
        status_info = st.empty()
        
        def update_progress(progress, info):
            progress_bar.progress(progress)
            
            if "status" in info and info["status"] == "complete":
                status_info.text("Optimization completed! Here are the results:")
            else:
                batch_info = f"Batch {info['batch']}/{info['total_batches']}"
                round_info = f"Round {info['round']}/{info['total_rounds']}"
                status_info.text(f"{batch_info} - {round_info}")
        
        outputs = optimizer.optimize_streamlit(inputs=dataset, p_extractor=p_extraction, rounds=rounds, callback=update_progress)
        
        status.update(label="😄 Optimization complete!", state="complete")
        st.markdown("<h5 style='text-align: left; white-space: nowrap;'>🎯 Optimization Result:</h5>", unsafe_allow_html=True)
        st.write(outputs)
