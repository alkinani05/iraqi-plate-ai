import requests
import time

url = "http://45.156.84.161:8501"
print(f"Waiting for {url} to come online...")

# Wait up to 20 minutes (120 * 10s)
for i in range(120):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            print("✅ Deployment Successful! App is Online.")
            break
    except:
        if i % 6 == 0:
            print(f"Time elapsed: {i*10}s - Still waiting...", end='\r', flush=True)
        time.sleep(10)
else:
    print("\n❌ Timed out waiting for app. Check VPS logs.")
