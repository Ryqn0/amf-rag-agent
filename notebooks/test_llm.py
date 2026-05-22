from amf_rag_agent import config
import anthropic

api_key = config.ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=api_key)

response = client.messages.create(
    max_tokens=1024,
    messages=[{
        "content": "What is the Autorité des Marchés Financiers (AMF) ?",
        "role": "user",
    }],
    model="claude-haiku-4-5-20251001",
)

print(response.content[0].text)