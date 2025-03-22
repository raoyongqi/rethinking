import requests

# 获取 Graphviz 下载页面的内容
url = "https://graphviz.gitlab.io/download/"
response = requests.get(url)

# 将网页内容保存到文件
with open("graphviz_download_page.html", "w", encoding="utf-8") as file:
    file.write(response.text)

print("网页已保存为 'graphviz_download_page.html'")
