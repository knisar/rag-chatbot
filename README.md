# RAG Chatbot using Langgraph and Tracing via W&B Weave

To get started:
* Setup `benchmark.env` in `./config` with necessary API keys (`WANDB_API_KEY`, `OPENAI_API_KEY`)
* Run the `langgraph_rag.py` file
  * Set the configs to your corresponding W&B projects and entity and `chat_model` if you want to use some other model

* Once the above file is run, you can start the streamlit app by running `streamlit run langgraph_bot.py` which will spin up a streamlit chat interface which you can use to interact with and all the traces will be logged to W&B Weave
   * Make sure to also set the configs in `langgraph_bot.py` file

