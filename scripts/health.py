import requests

r = requests.get("http://localhost:8000/health", timeout=5)
print(f"[{r.status_code}] {r.json()}")