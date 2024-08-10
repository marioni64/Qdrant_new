import os
import uuid
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct

qdrant_client = QdrantClient(
    url="",
    api_key="",
)
collection_name="Gyber_Collection"

qdrant_client.delete_collection(
    collection_name=collection_name
)


qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size= 1536, distance=models.Distance.COSINE),
)
info = qdrant_client.get_collection(collection_name="Gyber_Collection")


def read_data_from_pdf():
    pdf_path = '/qdrant/storage/data.pdf'
    text = ""

    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding(text_chunks, model_id="text-embedding-ada-002"):
    points = []
    for idx, chunk in enumerate(text_chunks):
        response = openai.Embedding.create(
            input=chunk,
            model=model_id
        )
        embeddings = response
        point_id = str(uuid.uuid4())

        points.append(PointStruct(
            id=point_id,
            vector=embeddings,
            payload={"text": chunk}))

    return points

def insert_data(get_points):

    operation_info = qdrant_client.upsert(
    collection_name="Gyber_Collection",
    wait=True,
    points=get_points
)
    return operation_info

def create_answer_with_context(query):
    embeddings = openai.embeddings_utils.get_embeddings(
        input=query,
        model="text-embedding-ada-002"
    )

    search_result = qdrant_client.search(
        collection_name="Gyber_Collection",
        query_vector=embeddings,
        limit=3
    )

    prompt = "Context:\n"
    for result in search_result:
        prompt += result.payload['text'] + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"

    print("----PROMPT START----")
    print(":", prompt)
    print("----PROMPT END----")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
             "role": "user",
             "content": prompt
             }
        ]
        )

    return completion.choices[0].message.content

def main():

    get_raw_text=read_data_from_pdf()
    chunks=get_text_chunks(get_raw_text)
    vectors=get_embedding(chunks)

    insert_data(vectors)
    question="What are some of the limitations of GPT-4?"
    answer=create_answer_with_context(question)
    print("Answer", answer, "\n")


if __name__ == '__main__':
    main()