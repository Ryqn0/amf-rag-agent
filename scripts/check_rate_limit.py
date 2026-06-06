import httpx

URL = "http://localhost:8000/ask"
HEADERS = {"X-API-Key": "dev-key-12345", "Content-Type": "application/json"}
BODY = {"question": "test"}


for i in range(12):

    r = httpx.post(URL, headers=HEADERS, json=BODY, timeout=60)
    print(f"Request {i+1}: Status code: {r.status_code}, Response: {r.json()}")