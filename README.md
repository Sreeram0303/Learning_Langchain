# 📚 LangChain Learning Hub

Welcome to my personal repository documenting everything I’ve learned while working with **LangChain** — a framework that bridges large language models with external data, tools, and APIs to build powerful LLM-driven applications.

Each folder in this repo represents a core module or capability of LangChain, containing clean, well-commented code examples that illustrate concepts in action. Whether you're starting from scratch or diving deeper into Retrieval-Augmented Generation (RAG), you'll find everything here to accelerate your learning.

---

## 🧠 What You'll Learn

This repo is a complete walkthrough of the LangChain ecosystem:

| 📁 Folder | 🧩 Key Concepts Covered |
|----------|-------------------------|
| `Prompt/` | Prompt engineering basics — static and dynamic prompts using `PromptTemplate`, `ChatPromptTemplate`, message roles (`system`, `user`, `ai`) |
| `ChatModels/` | How to work with different LLM providers like OpenAI, HuggingFace, and Anthropic using LangChain |
| `Embedded Models/` | Embedding generation using `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, and use-cases in semantic similarity |
| `Output_format/` | Controlling the format of LLM responses through prompt-level instructions |
| `Output_parsers/` | Parsing LLM outputs with `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, and `StructuredOutputParser` |
| `Runnables/` | Composing modular logic using `RunnableSequence`, `RunnableParallel`, `RunnableLambda`, and `RunnablePassthrough` |
| `Chains/` | Building single-input/multi-step chains and branching logic |
| `RAG/` | Implementing Retrieval-Augmented Generation (RAG): document loaders, text splitters, vector stores, and retrievers |

---

## 📂 Repository Structure

```
.
├── Chains/               # Chain compositions and logic
├── ChatModels/           # Working with chat-based LLMs
├── Embedded Models/      # Embeddings and similarity search
├── Output_format/        # Structuring LLM response output
├── Output_parsers/       # Parsing outputs into structured types
├── Prompt/               # Prompt engineering and templates
├── RAG/                  # Retrieval-Augmented Generation (Loaders, Splitters, VectorStores, Retrievers)
├── Runnables/            # Core runnable components and chaining logic
└── README.md             # This file
```

Each folder includes:
- Jupyter Notebooks or `.py` files with working examples
