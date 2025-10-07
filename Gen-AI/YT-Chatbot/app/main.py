import streamlit as st
import re
import sys
from pathlib import Path

# FIX: Add the PROJECT ROOT to Python path, not just the app folder
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up two levels: app -> YT-Chatbot
sys.path.append(str(project_root))

from ingest.fetch_transcript import fetch_transcripts
from ingest.chunking import chunk_transcripts

# Rest of your code remains the same...
user_input = st.text_input("Paste the link here:")

match = re.search(
    r"(?:youtube(?:-nocookie)?\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
    user_input
)

video_id = match.group(1) if match else None

# FIX: Store the video_id in session state properly
if video_id:
    st.session_state["video_id"] = video_id

if st.button("Send"):
    video_id = st.session_state.get("video_id")
    if not video_id:
        st.error("Please paste a valid YouTube URL!")
    else:
        # Step 1: Fetch transcript
        with st.spinner("Fetching transcript..."):
            print("Step 1 - Starting to fetch transcript..........")
            transcript_text = fetch_transcripts(video_id)
        if not transcript_text:
            st.error("Transcript not available ❌")
        else:
            st.success("Transcript fetched ✅")
            st.session_state["transcript"] = transcript_text
            print("Transcript fetched successfully ✅")

            # Step 2: Chunk transcript
            with st.spinner("Chunking transcript..."):
                print("Starting to chunk transcript..........")
                chunks = chunk_transcripts(transcript_text)

            st.success(f"Transcript chunked into {len(chunks)} pieces ✅")
            st.session_state["chunks"] = chunks
            print("Transcript chunked successfully ✅")
            