import requests
import streamlit as st


def ask_query(question):
    response = requests.get(f"http://localhost:1457/assistant/{question}")
    response = response.json()["response"]
    response_text = response["response"]

    response_file_and_page = [
        (
            node["node"]["metadata"]["file_name"],
            node["node"]["metadata"]["page_label"],
            node["node"]["text"],
        )
        for node in response["source_nodes"]
    ]

    return response_text, response_file_and_page


def main():
    st.title("Text-to-SQL Paper Assistant")

    with st.chat_message("assistant"):
        st.write("Hello 👋 Ask a question about text-to-sql!")

    user_question = ""
    user_question = st.chat_input(
        placeholder="Can you give me examples of multi-task training for text-to-sql?"
    )

    if user_question != "" and user_question is not None:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Executing..."):
                response_text, response_file_and_page = ask_query(user_question)
                st.write(response_text)
                st.write("### References:")
                for file_name, page_label, passage in response_file_and_page:
                    st.markdown(
                        f">> File: {file_name} (page: {page_label})", help=passage
                    )


if __name__ == "__main__":
    main()
