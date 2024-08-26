import requests


url = "http://localhost:5000/query"

input_data = {"query": "Support vector machine ?"}


response = requests.post(url,json=input_data)

print(response.json())