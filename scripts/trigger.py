import requests

r = requests.post("http://localhost:8000/api/v1/trigger-close", timeout=10)
data = r.json()
print(f"[{r.status_code}] run_id={data['run_id']} status={data['status']}")
print(f"\nWatch worker: docker compose logs -f worker")
print(f"Inspect Redis: docker compose exec redis redis-cli KEYS 'close:{data['run_id']}:*'")