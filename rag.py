# 调用llm
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).quantize(4).half().cuda()  # 量化
#model = AutoModel.from_pretrained("model/chatglm3-6b", trust_remote_code=True).half().cuda()
movel = model.eval()

# 定义文件路径
filepath = "sidamingzhu.txt"

# 加载文件
from langchain_community.document_loaders import TextLoader
loader = TextLoader(filepath)
docs = loader.load()
#print("docs:", docs)

# 文本分割
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=30, chunk_overlap=5, separator='。')
docs_split = text_splitter.split_documents(docs)
#print("docs_split:", docs_split)

# 调用embedding模型
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
embedding_func = SentenceTransformerEmbeddings(model_name="model/text2vec-base-chinese")

# 构建向量库
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(docs_split, embedding_func)
vector_store.save_local("faiss_index")  # 保存索引
vector_store = FAISS.load_local("faiss_index", embedding_func, allow_dangerous_deserialization=True)  # 加载索引

# 记忆模块
from FlagEmbedding import FlagLLMReranker
from langchain.memory import ConversationBufferWindowMemory
#memory = ConversationSummaryMemory(llm=model)
memory = ConversationBufferWindowMemory(k=5)

history = []
while True:
    query = input("用户：")
    # 根据提问匹配上下文
    docs = vector_store.similarity_search_with_score(query)
    #print("docs:", docs)
    context = []
    for doc in docs:
        if doc[1] < 330:   # 文本相似度阈值
            context.append(doc[0].page_content)
    print("context:", context)
    
    # 构造prompt
    prompt = f"已知信息：\n{context}\n根据已知信息回答问题：\n{query}"
    
    # llm 生成回答
    response, history = model.chat(tokenizer, prompt, history)
    
    print(response)
    # memory.save_context({"input": query}, {"output": response[0]})
    # print(memory.load_memory_variables({}))