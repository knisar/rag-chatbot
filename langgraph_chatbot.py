import streamlit as st
import weave
from weave import Scorer
import asyncio
import uuid
from typing import List, Dict, Any
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import time
import cProfile
import pstats
from pydantic import BaseModel, Field

# Download required NLTK data
nltk.download('punkt')

# Configuration dictionary
config = {
    "entity": "wandb-smle",
    "project_name": "langgraph-demo1",
    "model_name": "gpt-4-turbo",
    "temperature": 0.7,
    "collection_name": "climate_solutions",
    "persist_directory": "chroma_db",
    "model_url": "weave:///wandb-smle/langgraph-demo1/object/rag_model:latest",
    "prompt_template": """You are a helpful AI assistant. Use the following context to answer the question. 
        If the context is relevant, provide a detailed answer. If the context is not relevant, expand based on your knowledge.

        Context:
        {context}

        Question: {query}

        Answer:"""
}

@st.cache_resource(show_spinner=False)
def get_weave_model_and_components(config: Dict):
    """Cache both model and components to avoid reinitialization"""
    print("Initializing model and components...")
    model = weave.ref(config["model_url"]).get()
    components = model.initialize_components()
    return model, components

@weave.op()
def evaluate_rag_response(response: str, context: List[str]) -> Dict:
    """Evaluate RAG response using multiple metrics"""
    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    rouge_scores = []
    for reference in context:
        score = rouge.score(response, reference)
        rouge_scores.append({
            'rouge1': score['rouge1'].fmeasure,
            'rouge2': score['rouge2'].fmeasure,
            'rougeL': score['rougeL'].fmeasure
        })
    
    final_rouge_scores = {
        'rouge1': max(s['rouge1'] for s in rouge_scores) if rouge_scores else 0.0,
        'rouge2': max(s['rouge2'] for s in rouge_scores) if rouge_scores else 0.0,
        'rougeL': max(s['rougeL'] for s in rouge_scores) if rouge_scores else 0.0
    }
    
    # Calculate BLEU score
    prediction_tokens = word_tokenize(response.lower())
    reference_tokens = [word_tokenize(ref.lower()) for ref in context]
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        reference_tokens, 
        prediction_tokens,
        smoothing_function=smoothing,
        weights=(0.25, 0.25, 0.25, 0.25)
    )
    
    # Calculate hallucination score (without terms)
    pred_tokens = set(word_tokenize(response.lower()))
    context_tokens = set()
    for ctx in context:
        context_tokens.update(word_tokenize(ctx.lower()))
    
    common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'])
    pred_tokens = pred_tokens - common_words
    context_tokens = context_tokens - common_words
    
    overlap_tokens = pred_tokens.intersection(context_tokens)
    hallucination_score = len(overlap_tokens) / len(pred_tokens) if pred_tokens else 1.0
    
    return {
        'rouge_scores': final_rouge_scores,
        'bleu_score': bleu_score,
        'hallucination_score': hallucination_score,
        'context_length': len(' '.join(context)),
        'response_length': len(response)
    }

class RAGModel(weave.Model, BaseModel):
    """RAG model wrapper for Weave evaluation"""
    config: Dict = Field(default_factory=dict)
    model: Any = Field(default=None, exclude=True)
    components: Any = Field(default=None, exclude=True)
    
    def initialize(self, model, components):
        """Initialize with pre-loaded model and components"""
        self.model = model
        self.components = components

    @weave.op()
    async def predict(self, query: str) -> Dict:
        """Process a query through the RAG pipeline"""
        components = self.components
        
        try:
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(
                None, 
                lambda: components.vectorstore.similarity_search(query, k=2)
            )
            
            # Deduplicate context by using a set
            context = list(set(doc.page_content for doc in docs))
            
            prompt_with_context = self.config["prompt_template"].format(
                context='\n'.join(context),  # Changed from ' '.join to '\n'.join for better readability
                query=query
            )
            
            response = await self.model.predict(prompt_with_context)
            result_content = response["result"]["content"]
            
            return {
                "response": result_content,
                "context": context,
                "query": query,
                "output": {
                    "response": result_content,
                    "context": context,
                    "query": query
                }
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

@weave.op()
async def process_response(model: RAGModel, prompt: str) -> Dict:
    """Process response and get call object in a Weave operation"""
    response = await model.predict(prompt)
    current_call = weave.require_current_call()
    return {
        "response": response,
        "call": current_call
    }

class RAGScorer(Scorer):
    """Scorer for RAG evaluation metrics"""
    
    @weave.op
    def score(self, output: Dict, target: Dict) -> Dict:
        """Score the RAG output
        Args:
            output: The response from the RAG system
            target: The target containing context
        """
        metrics = evaluate_rag_response(
            response=output["response"],
            context=output["context"]
        )
        return metrics

def display_metrics(metrics: Dict):
    """Display metrics in a streamlit expander"""
    with st.expander("Response Metrics", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            st.metric("ROUGE-1", f"{metrics['rouge_scores']['rouge1']:.2f}")
        with cols[1]:
            st.metric("BLEU", f"{metrics['bleu_score']:.2f}")
        with cols[2]:
            st.metric("Hallucination", f"{metrics['hallucination_score']:.2f}")

def add_feedback_ui(call, key_suffix: str):
    """Add feedback UI elements"""
    st.markdown("Your feedback: ")
    feedback_text = st.text_input("", 
                               placeholder="Type your thoughts...",
                               key=f"note_{key_suffix}")
    st.button(":thumbsup:", 
             on_click=lambda: call.feedback.add_reaction("👍"), 
             key=f'up_{key_suffix}')
    st.button(":thumbsdown:", 
             on_click=lambda: call.feedback.add_reaction("👎"), 
             key=f'down_{key_suffix}')
    st.button("Submit Feedback", 
             on_click=lambda: call.feedback.add_note(feedback_text), 
             key=f'submit_{key_suffix}')

def display_chat_history():
    """Display existing chat messages with feedback UI"""
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if "call" in msg:
                    add_feedback_ui(msg["call"], idx)
                if "metrics" in msg:
                    display_metrics(msg["metrics"])

def process_new_message(model: RAGModel, prompt: str):
    """Process new message and display response"""
    with st.chat_message("assistant"):
        try:
            # Get response and call object within Weave operation
            with weave.attributes({'session': st.session_state['session_id']}):
                result = asyncio.run(process_response(model, prompt))
                response = result["response"]
                current_call = result["call"]
                result_content = response["response"]
                
                # Store in session state first
                msg_data = {
                    "role": "assistant", 
                    "content": result_content,
                    "context": response["context"],
                    "call": current_call
                }
                st.session_state.messages.append(msg_data)
                
                # Display response and feedback UI
                st.markdown(result_content)
                add_feedback_ui(current_call, 'new')
                
                # Add metrics
                evaluation = weave.Evaluation(
                    name='rag_evaluation',
                    dataset=[{
                        'query': prompt,
                        'output': {
                            'response': result_content,
                            'context': response["context"]
                        },
                        'target': {'context': response["context"][0]}
                    }],
                    scorers=[RAGScorer()]
                )
                
                eval_results = asyncio.run(evaluation.evaluate(model, __weave={"display_name": config["model_name"]}))
                metrics = eval_results["RAGScorer"][0]
                msg_data["metrics"] = metrics
                st.session_state.messages[-1] = msg_data
                display_metrics(metrics)
                
        except Exception as e:
            print(f"Error during processing: {e}")

def get_and_process_prompt(model: RAGModel):
    """Main function to handle chat interaction"""
    # Display existing chat history
    display_chat_history()
    
    # Handle new input
    if prompt := st.chat_input("What is up?", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process response
        process_new_message(model, prompt)

def main():
    st.title("RAG Chatbot with Evaluation")
    
    # Initialize Weave first
    weave.init(f'{config["entity"]}/{config["project_name"]}')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "model_components" not in st.session_state:
        try:
            model, components = get_weave_model_and_components(config)
            st.session_state.model_components = (model, components)
        except Exception as e:
            st.error(f"Error initializing model: {e}")
            return
    
    # Create RAG model instance with cached components
    try:
        model, components = st.session_state.model_components
        rag_model = RAGModel(config=config)
        rag_model.initialize(model, components)
        get_and_process_prompt(rag_model)
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        print(f"Initialization error: {e}")

if __name__ == "__main__":
    main()