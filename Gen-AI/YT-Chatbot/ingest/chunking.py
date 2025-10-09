from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcripts(transcript_text: str):
    """Split transcript text into smaller chunks."""
    if not transcript_text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " "],
    )
    chunks = text_splitter.split_text(transcript_text)
    print(chunks[0])
    return chunks
