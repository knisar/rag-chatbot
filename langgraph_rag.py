import os, yaml
import asyncio
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import weave
from typing import Any, Dict, TypedDict, Sequence, Union, List, Optional
import operator
from typing_extensions import Annotated
from dataclasses import dataclass

# LangChain imports
from langchain_core.globals import set_debug, set_verbose
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import Graph, END, START
from langgraph.prebuilt.tool_executor import ToolExecutor

# Other imports remain the same
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set LangChain configuration
set_verbose(False)
set_debug(False)

load_dotenv()

# Configuration
config = {
    "entity": 'wandb-smle',
    "project_name": 'langgraph-demo1',
    "chat_model": "gpt-4-turbo",
    "cm_temperature": 0.7,
    "source_data_path": "configs/raw_data.csv",
    # "model_ref": "weave:///knisar/langgraph-demo/object/rag_model:latest", # Uncomment and update after first run
    "rag_prompt_user": """
    Answer the question based on the context provided. 
    If you cannot find the answer in the context, say "I don't have enough information to answer this question."
    
    Context: {context}
    Question: {question}
    
    Answer: """
}

# Define state type
class AgentState(TypedDict):
    query: str
    context: list[str]
    response: str
    sources: list[Dict]

@dataclass
class RAGConfig:
    """Configuration for RAG model that can be serialized"""
    model_name: str
    temperature: float
    collection_name: str
    persist_directory: str
    prompt_template: str
    source_data_path: str
    version: str = "1.0.0"

@dataclass
class RAGComponents:
    """Runtime components that don't need to be serialized"""
    embeddings: OpenAIEmbeddings
    llm: ChatOpenAI
    vectorstore: Chroma
    template: ChatPromptTemplate
    documents_loaded: bool = False

class RAGModel(weave.Model):
    """RAG model with proper Weave integration"""
    config: RAGConfig
    
    def __init__(self, config_dict: Dict):
        # Convert dict config to RAGConfig
        config = RAGConfig(
            model_name=config_dict["chat_model"],
            temperature=config_dict["cm_temperature"],
            collection_name="climate_solutions",
            persist_directory="chroma_db",
            prompt_template=config_dict["rag_prompt_user"],
            source_data_path=config_dict["source_data_path"]
        )
        
        # Initialize with only the config
        super().__init__(config=config)

    @weave.op()
    def initialize_components(self):
        """Initialize or reinitialize components"""
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=embeddings,
            persist_directory=self.config.persist_directory
        )
        
        components = RAGComponents(
            embeddings=embeddings,
            llm=ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature
            ),
            vectorstore=vectorstore,
            template=ChatPromptTemplate.from_template(self.config.prompt_template)
        )
        
        # Load documents if needed
        if not components.documents_loaded:
            self.load_documents(components)
            components.documents_loaded = True
            
        return components

    @weave.op()
    def load_documents(self, components: RAGComponents):
        """Load documents into vectorstore"""
        df = pd.read_csv(self.config.source_data_path)
        documents = [
            Document(
                page_content=row['page_content'],
                metadata={"title": row['title'], "type": row['type']}
            )
            for _, row in df.iterrows()
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        components.vectorstore.add_documents(splits)

    @weave.op()
    async def predict(self, query: str) -> Dict:
        """Process a query through the RAG pipeline"""
        # Initialize components
        components = self.initialize_components()
        print(f"Initialized components for query: {query}")
        
        # Create graph components
        workflow = Graph()
        
        async def retrieve(state: AgentState) -> AgentState:
            """Async retrieval function"""
            docs = await asyncio.to_thread(
                components.vectorstore.as_retriever(
                    search_kwargs={"k": 2}
                ).get_relevant_documents,
                state["query"]
            )
            print(f"Retrieved {len(docs)} documents")
            if docs:
                print(f"First document content: {docs[0].page_content[:200]}")
            
            state["context"] = [doc.page_content for doc in docs]
            state["sources"] = [doc.metadata for doc in docs]
            return state

        async def generate(state: AgentState) -> AgentState:
            """Async generation function"""
            context = "\n\n".join(state["context"])
            print(f"Context length: {len(context)}")
            chain = (
                {"context": lambda x: context, "question": lambda x: state["query"]}
                | components.template
                | components.llm
                | StrOutputParser()
            )
            state["response"] = await asyncio.to_thread(chain.invoke, state["query"])
            print(f"Generated response: {state['response'][:200]}")
            return state

        # Setup graph
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Run prediction
        state = AgentState(
            query=query,
            context=[],
            response="",
            sources=[]
        )
        
        result = await workflow.compile().ainvoke(state)
        
        return {
            "result": {"content": result["response"]},
            "source_documents": result["sources"]
        }

@weave.op()
def create_rag_model(config: Dict) -> weave.ref:
    """Create and publish a new RAG model"""
    model = RAGModel(config)
    
    # Publish model
    ref = weave.publish(
        model,
        name="rag_model",
    )
    
    print(f"Model published to Weave with reference: {ref}")
    return ref

async def main(config: Dict):
    # Initialize Weave project
    weave.init(config["entity"] + "/" + config["project_name"])
    
    try:
        # Try to load existing model
        if "model_ref" in config and config["model_ref"]:
            print(f"Loading existing model from: {config['model_ref']}")
            model = weave.ref(config["model_ref"]).get()
        else:
            # Create new model
            print("Creating and publishing new model...")
            model_ref = create_rag_model(config)
            model = model_ref.get()
            print(f"New model created with reference: {model_ref}")
        
        # Run example questions
        questions = [
            "What are the main solutions for reducing food waste?",
            "How does forest protection help with climate change?",
            "What is the role of alternative refrigerants in reducing emissions?",
            "What is sustainable intensification for smallholders?"
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            response = await model.predict(question)
            print(f"\nAnswer: {response['result']['content']}")
            print("\nSources:")
            for doc in response["source_documents"]:
                print(f"- {doc['title']} ({doc['type']})")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


def get_weave_model(weave_url: str):
    model = weave.ref(weave_url).get()
    return model

if __name__ == "__main__":
    import asyncio
    # config = {}
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # with open(os.path.join(file_path, 'configs/general_config.yaml'), 'r') as file:
    #     config.update(yaml.safe_load(file))

    # if not load_dotenv(os.path.join(file_path, "configs/benchmark.env")):
    #     print("Environment variables couldn't be found.")
    asyncio.run(main(config))