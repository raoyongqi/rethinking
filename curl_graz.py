import requests

url = "https://graphviz.gitlab.io/download/"
response = requests.get(url)

with open("graphviz_download_page.html", "w", encoding="utf-8") as file:
    file.write(response.text)

print("网页已保存为 'graphviz_download_page.html'")
