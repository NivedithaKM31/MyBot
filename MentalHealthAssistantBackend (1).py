
from flask import Flask, request, jsonify
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp

# Initialize Flask app
app = Flask(__name__)

# Load vector store and LLM
retriever = ...  # Load your retriever (from previous code)
llm = LlamaCpp(
    model_path="/content/drive/MyDrive/BioMistral-7B.Q4_K_M.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1
)

template = """
<|context|>
You are Medical Assistant that follows the instructions and generates the accurate response based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define route for queries
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get("query", "")
    
    try:
        response = rag_chain.invoke(user_query)
        return jsonify({"answer": response})
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"answer": "Sorry, I couldn't process your query."}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
