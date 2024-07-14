# 第一部分 kg->向量库
username = "neo4j"
password = "41824144"
url = 'neo4j+s://46eba141.databases.neo4j.io'
embed_dim = 1536

from neo4j import GraphDatabase
driver = GraphDatabase.driver(url, auth=(username, password))
triple = []
# 遍历关系
result = driver.session().run("MATCH (s)-[p]->(o) RETURN s,p,o")
for record in result:
    triple.append([record["s"]['name'], record["p"].type, record["o"]['name']])

# 遍历节点
result = driver.session().run("MATCH (n) RETURN n")
for record in result:
    properties = record["n"]._properties
    for key, value in properties.items():
        if key != 'name':
            triple.append([record["n"]['name'], key, value])

driver.close()

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).quantize(4).half().cuda()  # 量化
model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()
knowledge = []
for tri in triple:
    prompt = f"Given triplets:\n{tri}\nPlease express it in a concise sentence, Your answer should not contain the original triplet, and do not use single or double quotation marks.\nfor example:\n['adam', 'son', 'mike']\n your answer should be:\n mike is adam's son"
    print(prompt)
    response, _ = model.chat(tokenizer, prompt, history=[])
    knowledge.append(response)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_model = SentenceTransformerEmbeddings(model_name="model/text2vec-base-chinese")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)  # 加载索引
#vector_store = FAISS.from_texts(knowledge, embedding_model)
#vector_store.save_local("faiss_index")  # 保存索引


#  第二部分 查询问题
print("input 'exit' to break")
history = []
while True:
    query = input("query:")
    if query == 'exit':
        break
    stage1 = vector_store.similarity_search(query, 15)
    for i in range(len(stage1)):
        stage1[i] = stage1[i].page_content
    print(stage1)
    
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('model/bge-reranker-v2-m3', use_fp16=True)
    stage2 = [[query, s] for s in stage1]
    scores = reranker.compute_score(stage2)
    for i in range(len(stage2)):
        stage2[i][0] = scores[i]
    stage2.sort(reverse=True)
    print(stage2)
    
    context = []  # 最终的上下文
    for st in stage2:
        if(st[0] < -8):
            break
        context.append(st[1])
    prompt = f"Given information:\n{context}\nAnswer the following question based on known information or prior knowledge:\n{query}"
    print(prompt)
    response, history = model.chat(tokenizer, prompt, history)
    print(response)

