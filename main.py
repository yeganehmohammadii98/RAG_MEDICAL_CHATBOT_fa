import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader
import openai

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key"


def load_documents():
    # Load documents from a directory
    loader = DirectoryLoader('./home/yeganeh/PycharmProjects/RAG-app/RAG_MEDICAL_CHATBOT_fa/data/', glob="**/*.json")
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    return texts


def create_vectorstore(texts):
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Save vectorstore locally
    vectorstore.save_local("faiss_index")
    return vectorstore


def load_vectorstore():
    # Load existing vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    return vectorstore


def create_chain(vectorstore):
    # Create retrieval chain
    llm = OpenAI(temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )
    return chain


def chat_with_bot():
    # Initialize conversation history
    chat_history = []

    # Load or create vectorstore
    if os.path.exists("faiss_index"):
        vectorstore = load_vectorstore()
    else:
        texts = load_documents()
        vectorstore = create_vectorstore(texts)

    # Create conversation chain
    chain = create_chain(vectorstore)

    # Chat loop
    while True:
        # Get user input
        question = input("You: ")
        if question.lower() in ['quit', 'exit', 'bye']:
            break

        # Get response from chain
        response = chain({"question": question, "chat_history": chat_history})

        # Update chat history
        chat_history.append((question, response['answer']))

        # Print response
        print(f"Bot: {response['answer']}\n")


if __name__ == "__main__":
    chat_with_bot()
