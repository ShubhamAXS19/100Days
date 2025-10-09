import streamlit as st
import re
import sys
from pathlib import Path

# ----------------------------------------------------------
# ✅ PATH SETUP: Ensure imports work even when using Streamlit
# ----------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # YT-Chatbot/
sys.path.append(str(project_root))

from ingest.fetch_transcript import fetch_transcripts
from ingest.chunking import chunk_transcripts
from ingest.embed_store import generate_embeddings, store_embeddings


# ----------------------------------------------------------
# ✅ STREAMLIT UI
# ----------------------------------------------------------
st.set_page_config(page_title="Chat with YouTube 🎥", layout="centered")
st.title("💬 Chat with a YouTube Video")

user_input = st.text_input("Paste your YouTube link here:")

# Extract video_id from URL
match = re.search(
    r"(?:youtube(?:-nocookie)?\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
    user_input
)
video_id = match.group(1) if match else None

if video_id:
    st.session_state["video_id"] = video_id


# ----------------------------------------------------------
# ✅ MAIN PIPELINE
# ----------------------------------------------------------
if st.button("Send"):
    video_id = st.session_state.get("video_id")

    if not video_id:
        st.error("❌ Please paste a valid YouTube URL.")
        st.stop()

    try:
        # ---------------- Step 1: Fetch Transcript ----------------
        with st.spinner("Fetching transcript..."):
            st.write("🔹 Step 1: Fetching transcript...")
            transcript_text = fetch_transcripts(video_id)

        if not transcript_text or len(transcript_text.strip()) == 0:
            st.error("❌ Transcript not available for this video.")
            st.stop()
        else:
            st.success("✅ Transcript fetched successfully.")
            st.session_state["transcript"] = transcript_text
            st.write(f"📜 Transcript length: {len(transcript_text)} characters")

        # ---------------- Step 2: Chunk Transcript ----------------
        with st.spinner("Chunking transcript..."):
            st.write("🔹 Step 2: Splitting transcript into chunks...")
            chunks = chunk_transcripts(transcript_text)

        if not chunks or len(chunks) == 0:
            st.error("❌ No chunks created. Check chunking logic or empty transcript.")
            st.stop()
        else:
            st.success(f"✅ Created {len(chunks)} chunks.")
            st.session_state["chunks"] = chunks

        # ---------------- Step 3: Generate Embeddings ----------------
        with st.spinner("Generating embeddings..."):
            st.write("🔹 Step 3: Generating embeddings with Gemini...")
            try:
                embeddings = generate_embeddings(chunks)
            except Exception as e:
                st.error(f"❌ Error during embedding generation: {e}")
                st.stop()

        if not embeddings or len(embeddings) != len(chunks):
            st.error("❌ Embedding mismatch — some chunks may not have been processed.")
            st.stop()
        else:
            st.success("✅ Embeddings generated successfully.")

        # ---------------- Step 4: Store in Pinecone ----------------
        with st.spinner("Storing embeddings in Pinecone..."):
            st.write("🔹 Step 4: Uploading embeddings to Pinecone index...")
            try:
                store_embeddings(embeddings)
            except Exception as e:
                st.error(f"❌ Failed to store embeddings in Pinecone: {e}")
                st.stop()

        st.success("🎉 All steps completed successfully! Embeddings are ready for retrieval.")
        st.balloons()

    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")
        st.stop()
