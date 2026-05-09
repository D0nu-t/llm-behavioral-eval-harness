from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="sk-lm-YvAwXyz9:5WOxwUhYTnIYbm1o4bz6",
)
model = "qwen2.5-0.5b-instruct"
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Ping"}],
)
try:
    print(response.choices[0].message.content)
except Exception as e:
    print("Error accessing response content:", e)
