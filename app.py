import streamlit as st

def main():
    st.set_page_config(page_title="Chat with PDFs")
    st.header("Chat with PDFs")
    st.text_input("Enter your question here:")

    with st.sidebar:
        st.subheader("Your files")
        st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        st.button("Enter")


if __name__ == '__main__':
    main()