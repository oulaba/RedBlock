import gensim.downloader as api

# 查看可用的预训练模型列表
# print(api.info())

# 下载并加载 "word2vec-google-news-300" 模型
model = api.load("word2vec-google-news-300")