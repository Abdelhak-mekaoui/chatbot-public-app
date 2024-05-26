# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import  pipeline, AutoTokenizer
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import torch
load_dotenv()


import nest_asyncio
nest_asyncio.apply()



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model locally
model_id = "TinyLlama-1.1B-Chat-v1.0"
model_path = f"./data/{model_id}"
cahce_path = './cache'

from transformers import LlamaForCausalLM  

model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"cache_dir": cahce_path, 'device_map': 'auto' , "torch_dtype" : torch.bfloat16, "max_length" : 2000},
                max_new_tokens=256,  do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

hf = HuggingFacePipeline(model_id=model_id, pipeline=pipe)
hf.pipeline.model._validate_model_kwargs = lambda self: None


uri = "mongodb://root:password@localhost:27017"
db="chat-docs"
collection="docs"

# Setup MongoDB loader
loader = MongodbLoader(
    connection_string=uri,#os.getenv('MONGO_URI'),
    db_name=db,#os.getenv('MONGO_DB'),
    collection_name=collection,#os.getenv('MONGO_COLLECTION'),
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the prompt using special tokens that your model understands
template = """<|system|>Repond selon le contexte : {context} </s> <|user|>{question} Repond en français </s> <|assistant|>"""
prompt = ChatPromptTemplate.from_template(template)

class ChatQuestion(BaseModel):
    chatQuestion: str

@app.post("/v1/")
async def create_item(item: ChatQuestion):

    print("processing ...")
    print(item)
    data = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(data, embeddings)
    retriever = vectorstore.as_retriever()

    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | hf
    | StrOutputParser()
    )    
    response = chain.invoke(item.chatQuestion)
    split_response = response.split(' <|assistant|>', 1)[-1]
    
    print("end processing")
    return {"chatResponse": split_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
