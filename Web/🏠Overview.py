import streamlit as st


p2p_star = {
    "not ( True ) and ( True ) is": "Use this statement with a conditional if know what is the value True of and what Not False means. Or not True and also boolean. In explain your reasoning.",
    "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =": "Use your knowledge reasoning and think step by step. Finally give the actual correct answer.",
    "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?": "Use the date today which will not would give us an error. solution is given as answer date is correct the option data and the current month and year to get to previous and current month of year to determine what the current data will look."
}

st.set_page_config(
    layout="wide",
    page_title="GreaterOptimizer",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem;
    }
    
    .main-header {
        display: flex;
        justify-content: center;
        text-align: center;
        padding: 0;
        margin: 0 auto;
        max-width: 100%;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin-top: 0px;
        margin-bottom: 0px;
        white-space: nowrap;
    }
    .author-text {
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 0px;
        white-space: nowrap;
    }
    .author-text a {
        text-decoration: none;
        color: inherit;
    }
    .author-text a:hover {
        text-decoration: underline;
        color: #4682B4;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 100px;
        margin-top: 10px;
    }
    .button-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .button-text {
        margin-top: 10px;
        color: #4682B4;
        font-weight: bold;
        font-size: 18px;
    }
    
    footer {
        visibility: hidden;
    }
    
    .overview-text {
        margin-top: 20px;
        text-align: justify;
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1 class="title-text">GreaterOptimizer: A Python Toolkit for Prompt Optimization</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="main-header"><h3 class="author-text">\
            <a href="mailto:wmz5132@psu.edu">Wenliang Zheng</a>,\
            <a href="mailto:sfd5525@psu.edu">Sarkar Snigdha Sarathi Das</a>,\
            <a href="mailto:rmz5227@psu.edu">Rui Zhang</a>\
            </h3></div>', unsafe_allow_html=True)

st.markdown("""
<div class="button-container">
    <div class="button-item">
        <a href="https://arxiv.org/pdf/2412.09722" target="_blank" style="text-decoration: none;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/PDF_file_icon.svg" width="80" height="80">
            <div class="button-text">Paper</div>
        </a>
    </div>
    <div class="button-item">
        <a href="https://github.com/WenliangZhoushan/GreaterPrompt" target="_blank" style="text-decoration: none;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="80" height="80">
            <div class="button-text">Code and Data</div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="overview-text">
<p><strong>Overview of Research</strong>: The performance of large language models (LLMs) is significantly influenced by prompt design, making prompt optimization a crucial area of study. Traditional methods for optimizing prompts heavily depend on textual feedback from large, closed-source models like GPT-4, which analyze inference errors and suggest refinements. However, this reliance on computationally expensive LLMs limits the efficiency of smaller, open-source models that lack the ability to generate high-quality optimization feedback on their own.</p>

<p>This research introduces GReaTer, a novel prompt optimization technique that leverages gradient information over reasoning to enhance prompt effectiveness for smaller LLMs without relying on external, proprietary models. Unlike prior approaches that operate purely in the text space, GReaTer utilizes task loss gradients, allowing direct optimization of prompts. This method empowers smaller, open-source models to achieve state-of-the-art performance without assistance from larger LLMs.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    see = st.markdown("<h3>👀 See Examples</h3>", unsafe_allow_html=True)
    questions = st.selectbox("Select a question", [
        "not ( True ) and ( True ) is", "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =", "Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?"
    ])
    p_init = st.text_input(label="Initial P", value="Use logical reasoning to think it.", disabled=True)
    p_star = st.text_area(label="Optimized P*", value=p2p_star[questions], height=200, disabled=True)

st.write("<h2 style='text-align: center; white-space: nowrap;'>🤗 Now pick a method on the left side bar to get started!</h2>", unsafe_allow_html=True)
