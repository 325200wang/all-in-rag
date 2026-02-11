import os
# 如果在国内使用 Hugging Face，可通过设置镜像端点来加速/适配
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv

# LlamaIndex 主要类和工具
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# OpenAILike：适配类，用来把类似 OpenAI 的 API（或兼容的服务）接入 LlamaIndex
from llama_index.llms.openai_like import OpenAILike
# HuggingFaceEmbedding：使用 Hugging Face 模型构建向量嵌入
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ----------------------------
# 配置 LLM（模型）和 Embedding 模型
# ----------------------------
# 使用 AIHubmix（示例）作为后端，OpenAILike 作为一个兼容层来适配 LlamaIndex 的调用方式
Settings.llm = OpenAILike(
    model="glm-4.7-flash-free",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://aihubmix.com/v1",
    is_chat_model=True
)

# 如果需要切换到其它兼容的服务，可以参考下面的注释示例
# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )

# 指定用于生成向量嵌入的模型，这里使用 BAAI 的中文小模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# ----------------------------
# 读取文档并构建索引
# ----------------------------
# SimpleDirectoryReader 用于从文件中读取文档（此处直接传入单个文件路径）
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 从文档构建向量索引。VectorStoreIndex 会使用上面配置的 embed_model 来生成向量
index = VectorStoreIndex.from_documents(docs)

# 将索引包装成查询引擎，提供统一的查询接口
query_engine = index.as_query_engine()

# get_prompts() 可以用来查看查询时使用的提示词模板，便于调试或自定义提示
print(query_engine.get_prompts())

# 执行一次示例查询并打印结果（返回的是 LlamaIndex 直接的查询结果对象/文本）
print(query_engine.query("文中举了哪些例子?"))