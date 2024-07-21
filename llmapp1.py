import streamlit as st
from huggingface_hub import login
from langchain import HuggingFaceHub
import pickle

from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain


huggingfacehub_api = "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"
login(huggingfacehub_api)


def model_choice(api_key):
    model_types = {
        "Llama 3":"meta-llama/Meta-Llama-3-8B-Instruct", 
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        }
    with st.container():
        model_name = st.selectbox(label="Choose your model!",options=list(model_types.keys()), placeholder = "HF model",index=None)
        if model_name:
            model_id=model_types[model_name]
            return HuggingFaceHub(huggingfacehub_api_token=api_key,
                                repo_id=model_id,
                                model_kwargs={"max_new_tokens": 256})#add more params here


def load_memory():
    #("Load memory from a pickle file.")
    try:
        with open("chat_memory.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def save_memory(memory):
    """Save memory to a pickle file."""
    with open("chat_memory.pkl", "wb") as f:
        pickle.dump(memory, f)

def clear_pickle_file():
    """Clear the pickle file by overwriting it with an empty ConversationBufferMemory."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    save_memory(memory)


def architecture():
    model = model_choice(huggingfacehub_api)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
    if prompts := st.chat_input(placeholder="Type your prompt here!"):
        st.session_state.messages.append({"role": "user", "content": prompts})

        # Code for prompt template and implementing memory
        prompt_t = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a chatbot having a conversation with a human."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )
        memory = load_memory()  # Load memory from pickle file
        chat_llm_chain = LLMChain(
            llm=model,
            prompt=prompt_t,
            verbose=True,
            memory=memory,
        )

        response = chat_llm_chain.predict(human_input=prompts)
        content = response.strip().split("Assistant: ")[-1].strip()

        st.session_state.messages.append({"role": "assistant", "content": content})
        

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        display_history(prompts, content)
        save_memory(memory)  # Save memory to pickle file


def display_history(user_input,response):
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "bot", "content": response})

    for entry in st.session_state.history:
        if entry["role"] == "user":
            user_input_cap = entry["content"].capitalize()
            st.sidebar.expander(label=user_input_cap).write(st.session_state.history[st.session_state.history.index(entry) + 1]["content"])

    return st.sidebar.write("Added to history!")



def initialize_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "input" not in st.session_state:
        st.session_state.input = ""

def clear_chat_history():
    st.sidebar.title("**Chat History**")
    
    with st.sidebar.container():
    # Add a button inside the container
        if st.button("Clear"):
            st.session_state.history.clear()
            st.session_state.history = []
            st.session_state['input'] = ""  # Clears input field as well
            st.sidebar.error("*Chat history is empty.*")
            #clear_pickle_file()
            st.rerun() 

def main():
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit text-generating chatbot")
    initialize_history()
    clear_chat_history()
    architecture()
    
    
if __name__ == "__main__":
    main()
