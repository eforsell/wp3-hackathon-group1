from langchain_core.prompts import ChatPromptTemplate

# Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistant for the employee catalog."),
        ("user", "{input}"),
    ]
)
