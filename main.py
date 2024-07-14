"""
from llama_index.llms.ollama import Ollama
llm = Ollama(model="model/chatglm3", request_timeout=30.0)
# from llama_index.legacy.llms import HuggingFaceLLM
# llm = HuggingFaceLLM(
#     model_name="model/chatglm3-6b",
#     tokenizer_name="model/chatglm3-6b",
#     #query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
#     context_window=3900,
#     max_new_tokens=256,
#     #model_kwargs={"quantization_config": quantization_config},
#     # tokenizer_kwargs={},
#     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
#    # messages_to_prompt=messages_to_prompt,
#     device_map="auto",
# )

from llama_index.legacy.vector_stores.neo4jvector import Neo4jVectorStore
username = "neo4j"
password = "41824144"
url = 'neo4j+s://46eba141.databases.neo4j.io'
embed_dim = 1536
neo4j_vector = Neo4jVectorStore(username, password, url, embedding_dimension=embed_dim)
from llama_index.core.storage.storage_context import StorageContext
storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
print(storage_context)
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
    llm=llm
)
"""

# 加载本地embeeding模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embedding_model = HuggingFaceEmbedding(model_name="model/text2vec-base-chinese")

# 读取文件夹，构建向量库index
from llama_index.core import GPTVectorStoreIndex, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
documents = SimpleDirectoryReader('knowledge').load_data()
index = GPTVectorStoreIndex.from_documents(documents, embed_model=embedding_model, transformations=[SentenceSplitter(chunk_size=100, chunk_overlap=10, separator="。")])

query = "唐僧为什么生孙悟空的气"
# 向量库检索
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve(query)
# query_engine = index.as_query_engine(llm=llm)
# print(query_engine.query("武大郎"))

# 展示检索节点
from llama_index.core.response.pprint_utils import pprint_source_node
for idx, result in enumerate(nodes):
        pprint_source_node(result)

# 加载rerank模型
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import QueryBundle
reranker = FlagEmbeddingReranker(top_n=3, model="model/bge-reranker-v2-m3")
query_bundle = QueryBundle(query_str=query)
reranked_nodes = reranker._postprocess_nodes(nodes, query_bundle=query_bundle)
for idx, result in enumerate(reranked_nodes):
        pprint_source_node(result)

# create index
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embedding_model)