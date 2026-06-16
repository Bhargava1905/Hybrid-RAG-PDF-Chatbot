import streamlit as st
from rag_chain import get_rag_chain
from upload_utils import process_uploaded_pdf
import os



# -----------------------------------
# SESSION STATE
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_pdf" not in st.session_state:
    st.session_state.selected_pdf = "All Documents"

if "last_selected_pdf" not in st.session_state:
    st.session_state.last_selected_pdf = "All Documents"

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📄"
)

st.title("Chat with your PDF")
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    if not os.path.exists(save_path):

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(
            "Processing PDF..."
        ):
            process_uploaded_pdf(
                save_path
            )

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )

        st.session_state.selected_pdf = uploaded_file.name

        st.cache_resource.clear()

        st.rerun()

pdf_files = [
    f for f in os.listdir("data")
    if f.endswith(".pdf")
]

options = ["All Documents"] + pdf_files

default_index = 0

if st.session_state.selected_pdf in options:
    default_index = options.index(
        st.session_state.selected_pdf
    )

selected_source = st.selectbox(
    "Select Document",
    options,
    index=default_index
)

if selected_source != st.session_state.last_selected_pdf:

    st.session_state.messages = []

    st.session_state.last_selected_pdf = selected_source
    
st.session_state.selected_pdf = selected_source

if selected_source == "All Documents":
    st.info("Searching across all documents")
else:
    st.info(f"Searching only in: {selected_source}")

# -----------------------------------
# LOAD RAG CHAIN
# -----------------------------------

@st.cache_resource
def load_chain():
    return get_rag_chain()
chain = load_chain()

# -----------------------------------
# DISPLAY CHAT HISTORY
# -----------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if (msg["role"] == "assistant"and "citations" in msg):

            st.markdown("Sources:")

            for citation in sorted(
                set(msg["citations"])
            ):
                st.markdown(
                    f"- {citation}"
                )

        # Show retrieved docs for assistant messages
        if msg["role"] == "assistant" and "docs" in msg:

            with st.expander("Retrieved Chunks"):

                for i, doc in enumerate(msg["docs"]):

                    st.markdown(f"### Chunk {i+1}")

                    source = doc.metadata.get("source","Unknown Document").replace(".pdf", "")

                    page = doc.metadata.get("page",0) + 1

                    st.markdown(f"**Source:** {source}")

                    st.markdown(f"**Page:** {page}")

                    # Chunk content
                    st.write(doc.page_content)

                    st.divider()


# -----------------------------------
# USER INPUT
# -----------------------------------

if question := st.chat_input("Ask a question about your PDF..."):

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
            history = ""

            for msg in st.session_state.messages[-6:]:

                history += (
                    f"{msg['role']}: "
                    f"{msg['content']}\n"
                )

            source_filter = None

            if selected_source != "All Documents":
                source_filter = selected_source

            # Invoke RAG pipeline
            result = chain(question,history,source_filter)

            answer = result["answer"]
            docs = result["docs"]
            citations = result["citations"]

        # Show answer
        st.markdown(answer)

        if citations:
            
            st.markdown("Sources:")

            for citation in sorted(set(citations)):
                st.markdown(
                    f"- {citation}"
                )

        # Show retrieved chunks
        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs):

                st.markdown(f"### Chunk {i+1}")

                source = doc.metadata.get("source","Unknown Document").replace(".pdf", "")

                page = doc.metadata.get("page",0) + 1

                st.markdown(f"**Source:** {source}")

                st.markdown(f"**Page:** {page}")

                st.write(doc.page_content)

                st.divider()

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "docs": docs,
        "citations": citations
    })