from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/shop/buy-iphone/iphone-16"

data = WebBaseLoader(url)
docs  = data.load()

print(docs)