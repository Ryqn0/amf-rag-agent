from amf_rag_agent import config
import anthropic

api_key = config.ANTHROPIC_API_KEY
client = anthropic.Anthropic(api_key=api_key)

def search_documents(query: str) -> list[str]:
    # Placeholder for document search logic
    return ["Fake chunk 1", "Fake chunk 2"]

tools = [
    {
        "name": "search_documents",
        "description": "Search for relevant documents based on a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    }
]


def run_agent(query: str, tools: list[dict]):

    history= [
        {
            "role": "user",
            "content": query,
        }
    ]

    done = False

    while not done:

        response = client.messages.create(model="claude-haiku-4-5-20251001",
                                          tools=tools,
                                          max_tokens=256, 
                                          messages=history)
    
        if response.stop_reason == "tool_use":
            
            tool_block = next(block for block in response.content if block.type == "tool_use")

            history.append(
                {
                    "role": "assistant",
                    "content": response.content
                }
            )

            result = search_documents(tool_block.input["query"])

            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": ("\n").join(result),
                        }
                    ],
                }
            )


        elif response.stop_reason == "end_turn":
            # model is done
            done = True
    
    return response.content[0].text

if __name__ == "__main__":
    answer = run_agent("What is the AMF?", tools)
    print(answer)