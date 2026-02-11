import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块：先演示默认分片行为，再用修改后的参数重新分片以便对比观察
# 1) 默认分片
text_splitter_default = RecursiveCharacterTextSplitter()
chunks_default = text_splitter_default.split_documents(docs)
print(f"默认分片数: {len(chunks_default)}，示例片段长度: {[len(c.page_content) for c in chunks_default[:2]]}")

# 2) 修改分片参数（以便观察 chunk_size 和 chunk_overlap 改变的效果）
#    可根据需要调整 chunk_size/chunk_overlap 观察不同结果
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"修改参数后分片数: {len(chunks)}，示例片段长度: {[len(c.page_content) for c in chunks[:2]]}")

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("AIHUBMIX_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

raw_answer = llm.invoke(prompt.format(question=question, context=docs_content))

# 尝试从不同可能的返回结构中提取模型的文本内容，只输出具体回答文本，过滤其他参数/元信息
def extract_content(ans):
    # 字符串直接返回
    if isinstance(ans, str):
        return ans
    # 字典常见结构
    if isinstance(ans, dict):
        if 'content' in ans:
            return ans['content']
        if 'text' in ans:
            return ans['text']
        if 'choices' in ans:
            parts = []
            for c in ans['choices']:
                if isinstance(c, dict):
                    if 'message' in c and isinstance(c['message'], dict) and 'content' in c['message']:
                        parts.append(c['message']['content'])
                    elif 'text' in c:
                        parts.append(c['text'])
            return '\n'.join(parts).strip()
    # 对象可能有 .content 或 .text 属性
    if hasattr(ans, 'content'):
        return getattr(ans, 'content')
    if hasattr(ans, 'text'):
        return getattr(ans, 'text')
    # 兜底
    return str(ans)

print(extract_content(raw_answer))
