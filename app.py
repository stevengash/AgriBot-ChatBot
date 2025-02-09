import os
import time
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchAPIWrapper
from langdetect import detect
from deep_translator import GoogleTranslator
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize AI Tools
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
duckduckgo_search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="in-en", time="y", max_results=2))

tools = [wiki, arxiv, duckduckgo_search]

# Initialize LLM using OpenAI's Library but Pointing to Groq
def load_llm():
    return ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=1,
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )

# Translate text to English
def translate_to_english(text):
    try:
        detected_lang = detect(text)  # Detect language
        if detected_lang == "en":
            return text, "en"  # No translation needed

        translated_text = GoogleTranslator(source=detected_lang, target="en").translate(text)
        return translated_text, detected_lang  # Return translated text and original language
    except Exception as e:
        return text, "unknown"  # Return original text if translation fails

# Translate text back to the original language
def translate_back(text, target_lang):
    try:
        if target_lang == "en":
            return text  # No translation needed

        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        return text  # Return original if translation fails

# Ensure Memory is Persistent Across Sessions
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create Conversational Agent with Proper Memory Usage
def get_conversational_agent():
    llm = load_llm()
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=st.session_state.chat_memory,
        verbose=True,
        return_intermediate_steps=False
        max_iterations=5,
        handle_parsing_errors=True
    )

# Streamlit Chat UI
def main():
    # Set Background Image
    page_bg_img = '''
    <style>
    .stApp {
        background: url("https://en.reset.org/app/uploads/2020/06/india_farming.jpg") no-repeat center center fixed;
        background-size: cover;
        background-color: rgba(0, 50, 0, 0.3); /* Dark green overlay with less transparency */
        background-blend-mode: darken; /* Ensures text remains readable */
    }
    .stChatMessage {
        background: rgba(0, 0, 0, 0.9); /* Light background for better contrast */
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: black; /* Ensures text is dark for readability */
    }
</style>

    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title("🌾 Agri Bot (Multilingual) 🌾")
    st.subheader("Your Smart Assistant for Farming and Agriculture")

    if st.button("Reset Conversation"):
        st.session_state.chat_memory.clear()
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    selected_language = st.sidebar.selectbox("Select Language:", list(available_languages.keys()), index=0)  # Default English
    language_code = available_languages[selected_language]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # Get user input
    prompt = st.chat_input("Ask your farming-related question here (in any language)...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            translated_query, original_lang = translate_to_english(prompt)

            st.write(f"🔍 *Detected Language:* {original_lang.upper()}")  # Show detected language
            st.write(f"🔄 *Translated Query:* {translated_query}")  # Show translated query

            agent = get_conversational_agent()

            def trim_chat_memory(max_length=5):#
                """ Retains only the last `max_length` messages in memory. """
                chat_history = st.session_state.chat_memory.load_memory_variables({})["chat_history"]
                if len(chat_history) > max_length:
                    st.session_state.chat_memory.chat_memory.messages = chat_history[-max_length:]#
                return chat_history

            # Apply trimming before invoking the agent
            chat_history = trim_chat_memory(max_length=5)#

            conversation_context = "\n".join([msg.content for msg in chat_history])

            full_prompt = f"""
                Previous conversation:
                {conversation_context}

                User: {prompt}

                Assistant: Think carefully. You are allowed to search a maximum of 2 times strictly.
                If you have found enough information from previous searches, STOP searching and generate an convincing answer using the available data.
                """

            # Retry in case of rate-limit errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = agent.invoke({"input": full_prompt})
                    break  # Exit loop if successful
                except Exception as e:
                    st.warning(f"⚠ API Rate Limit! Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(2)  # Wait and retry

            response_text = response["output"] if isinstance(response, dict) and "output" in response else str(response)
            final_response = translate_back(response_text, original_lang)  # Translate back to original language

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
