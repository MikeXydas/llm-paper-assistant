import streamlit as st
import requests



def ask_query(question):
    response = requests.get(f"http://localhost:1457/assistant/{question}")
    return response.json()["response"]


def main():
    st.title('Text-to-SQL Paper Assistant')

    with st.chat_message("assistant"):
        st.write("Hello 👋 Ask a question about text-to-sql!")


    user_question = ""
    user_question = st.chat_input(placeholder="Can you give me examples of multi-task training for text-to-sql?")

    
    if user_question != "" and user_question is not None:
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner('Executing...'):
                st.write(ask_query(user_question))


if __name__ == "__main__":
    main()