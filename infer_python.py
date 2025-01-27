import requests

url = "http://127.0.0.1:8085/predictions/vit-model"
file_path = "test.jpg"

headers = {
    'Content-Type': 'application/octet-stream'  # You might need to adjust this if it's not correct.
}

with open(file_path, 'rb') as f:
    response = requests.post(url, data=f, headers=headers)

print(response.status_code)
print(response.text)
