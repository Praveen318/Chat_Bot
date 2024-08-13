from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

txt_model = genai.GenerativeModel('gemini-pro')
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
def articulation_messages_func(question,vectordb_dir):
    

    persist_directory = vectordb_dir
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectordb = Chroma(
        persist_directory=persist_directory,    
        embedding_function=gemini_embeddings
    )

    def context_func(question,k=3, fetch_k=5):
        context = vectordb.max_marginal_relevance_search(question,k, fetch_k)
        # context = vectordb.similarity_search(question,k)
        return context
    context_docs = context_func(question,3,4)
    # print(context_docs)
    context = ' '.join(d.page_content for d in context_docs)
    context_sources = '\n'.join(str(d.metadata) for d in context_docs)
    # print(context_sources)
    articulation_messages = [{"role": "system", "content": """ A user will give you a context and a question. 
            Answer the question only based on the facts in the given context. 
            Ensure that the answer is relevant to the question.
            Avoid sensitive / controversial subjects.
            Don't share confidential information.
            Maintain polite and neutral language.
            If the question is ambiguous, or you are not sure how the question can be answered by the context, politely ask the user to rephrase the question.'
            """
            }, \
                    {"role": "user", "content": "context: " + str(context) + " question: " + str(question)}]
    print(context)
    # if os.path.exists(persist_directory):
    #     shutil.rmtree(persist_directory)
    #     print(f"Deleted existing directory: {persist_directory}")
    return articulation_messages, context_sources

            # If the question cannot be answered correctly using the context then say 'Sorry, no relevant context provided'.
            # If no context is provided then answer the question without using context.

# question = input("Enter your question: \n")
# articulation_messages, context_sources = articulation_messages_func(question)
# response = txt_model.generate_content(f"{articulation_messages}")
# # print(response.text)
# response = response.text + "\n Sources: \n" + context_sources

# print(response)