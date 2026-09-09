from app.chunking import chunk_pages


sample_pages = [
    {
        "page": 1,
        "text": """
        Organizational change refers to the process of modifying
        existing organizational structures, strategies, processes,
        technologies, or culture to improve organizational performance.
        """
    },
    {
        "page": 2,
        "text": """
        Resistance to change occurs when employees are unwilling
        or unable to accept changes in the organization. Communication,
        participation, training, and support can help reduce resistance.
        """
    }
]


chunks = chunk_pages(
    sample_pages,
    chunk_size=100,
    chunk_overlap=20
)


print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("\n--------------------")
    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Page    : {chunk['page']}")
    print(f"Text    : {chunk['text']}")