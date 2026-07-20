import json
import os
import signal
import subprocess
import sys
import time
from pathlib import  Path
import streamlit as st
from utils import remove_ansi_codes
#remove top-padding
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    model_name=st.selectbox("Model",
                       ["lr","dnn","deepfm","wide_deep","dcn"],
                       index=0
                       )
    train_data_path=st.text_input(label="train_data_path",value="~/.deepsklearn/data/criteo/debug_train.csv")
    validation_data_path=st.text_input(label="validation_data_path",value="~/.deepsklearn/data/criteo/debug_validation.csv")
    feature_config_path=st.text_input("feature_config_path",value="~/.deepsklearn/data/criteo/criteo_feature_column.json")
    label_config_path=st.text_input("feature_config_path",value="~/.deepsklearn/data/criteo/criteo_label_column.json")
    train_batch_size = st.number_input("train_batch_size", value=1000)
    validation_batch_size=st.number_input("validation_batch_size",value=1000)
    device = st.text_input("device", 'cpu')
    model_path=st.text_input(label="model_path",value=f"~/.deepsklearn/data/criteo/{model_name}/")

st.title("deepsklearn training dashboard")
left,center1,center2,right=st.columns([1,1,1,1])
with center1:
    start_training=st.button("start_training",
                             type="primary",
                             disabled=st.session_state.get("train_started",False)
                             )
with center2:
    stop_training=st.button("stop_training", type="primary")
st.subheader("train config")
train_config={
    "model_name":model_name,
    "train_data_path":train_data_path,
    "validation_data_path":validation_data_path,
    "feature_config":feature_config_path,
    "label_config":label_config_path,
    "train_batch_size":train_batch_size,
    "validation_batch_size":validation_batch_size,
    "model_path":model_path,
    "device":device
}
st.json(train_config)
config_path=f"../runtime_configs/{model_name}.json"
log_path = f"../logs/{model_name}.log"
if start_training:
    st.session_state["train_started"]=True
    # same file name will overwrite
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    cmd=[
        sys.executable,#current python commond
        "-u",# unbufferd mode
        "../scripts/train_non_sequence.py",
        "--config",
        config_path
    ]
    #start sub process to run the train code
    with open(log_path,"a",encoding="utf-8",buffering=1) as log_file:
        process=subprocess.Popen(cmd,
                                 stdout=log_file,
                                 stderr=log_file
                                 )
        print(f"traning process id:{process.pid}")
        st.session_state["train_pid"]=process.pid

if stop_training:
    st.session_state["train_started"]=False
    if "train_pid" not in st.session_state:
        st.warning("no training process found")
    else:
        try:
            train_pid=st.session_state["train_pid"]
            os.kill(train_pid,signal.SIGKILL)# kill -9 train_pid
            st.info(f"successfully kill pid {train_pid}")
            del st.session_state["train_pid"]
        except:
            train_pid = st.session_state["train_pid"]
            st.error(f"kill -9 {train_pid} faild")

if Path(log_path).exists():
    with open(log_path,"r",encoding="utf-8") as f:
        log_text=f.readlines() #return list
        if len(log_text)<1:
            log_text="waiting for training log"
        else:
            log_text="".join(log_text)
else:
    log_text = "waiting for training log"

st.subheader("training_log")
log_text=remove_ansi_codes(log_text)
st.text_area(label="log",
             value=log_text[-1000:],
             height=500,
             )
time.sleep(3)
st.rerun()


