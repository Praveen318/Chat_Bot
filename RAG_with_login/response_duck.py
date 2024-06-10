from duckduckgo_search import DDGS


def articulation_messages_duck_func(prompt):
    results = DDGS().text(prompt, max_results=5)  

    # print(context_docs)
    context = ' '.join(result['body'] for result in results)
    context_sources = '\n'.join(result['href'] for result in results)
    # print(context_sources)
    articulation_messages = [{"role": "system", "content": """ A user will give you a context and a question. 
            Summarize the context according to the question. 
            """
            }, \
                    {"role": "user", "content": "context: " + str(context) + " question: " + str(prompt)}]
    print(context)
    # if os.path.exists(persist_directory):
    #     shutil.rmtree(persist_directory)
    #     print(f"Deleted existing directory: {persist_directory}")
    return articulation_messages, context_sources