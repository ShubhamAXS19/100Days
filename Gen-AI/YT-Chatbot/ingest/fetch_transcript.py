from langchain_community.document_loaders import YoutubeLoader

def fetch_transcripts(video_id: str):
    """Fetch transcript for a given YouTube video ID using LangChain."""
    try:
        # Create YouTube URL from video ID
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Load transcript using LangChain
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info=False  # Set to True if you want metadata
        )
        
        # Load the documents
        documents = loader.load()
        
        # Extract text from documents
        if documents:
            all_text = " ".join([doc.page_content for doc in documents])
            print(f"Transcript loaded successfully! Length: {len(all_text)} characters")
            return all_text
        else:
            print("No transcript found")
            return None
            
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None