from typing import Tuple

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@st.cache_resource(show_spinner=False)
def get_bert_fill_mask():
    return pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def get_gpt2_generator():
    return pipeline("text-generation", model="gpt2")


@st.cache_resource(show_spinner=False)
def get_gpt2_lm_and_tokenizer(device: torch.device) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    return model, tokenizer
