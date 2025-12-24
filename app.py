import os
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from main import app
import pypdf
from auth import authenticate_user
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pdfminer.high_level import extract_text as pdfminer_extract_text

# --- 1. SETUP DATABASE ---
conn_string = "sqlite+aiosqlite:///history.db"
try:
    storage = SQLAlchemyDataLayer(conninfo=conn_string)
except Exception as e:
    print(f"⚠️ Database Error: {e}")
    storage = None

@cl.data_layer
def get_data_layer():
    return storage

# --- 2. AUTHENTICATION ---
@cl.password_auth_callback
def auth(username, password):
    return authenticate_user(username, password)

# --- 3. PINECONE & OPTIMIZER SETUP ---
llm_optimizer = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    embeddings = None
    print(f"⚠️ Warning: Could not load Embeddings model: {e}")

INDEX_NAME = os.environ.get("PINECONE_INDEX", "agent-memory")

def optimize_query(user_query: str) -> str:
    system_prompt = """
You are a Search Query Optimizer.
Convert the user's request into document-matching KEYWORDS.

Rules:
1) If user asks for summary, output:
"Abstract, Introduction, Conclusion, Key Findings, Overview"
2) If user asks a specific question, extract key technical terms.
3) Output ONLY keywords, no explanation.
""".strip()

    try:
        optimized = llm_optimizer.invoke(f"{system_prompt}\nUser Query: {user_query}").content.strip()
        print(f"    ✨ Optimized Query: '{user_query}' -> '{optimized}'")
        return optimized or user_query
    except Exception as e:
        print(f"    ⚠️ Optimization failed: {e}")
        return user_query

def ingest_to_pinecone(text_content: str):
    if not embeddings:
        raise RuntimeError("Embeddings model not available.")

    text_content = (text_content or "").strip()
    if len(text_content) < 200:
        raise RuntimeError("Extracted text is too short. PDF might be scanned or unreadable.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = splitter.create_documents([text_content])

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    return True

def vector_search(user_query: str) -> str:
    if not embeddings:
        return "Vector Search Unavailable: embeddings model not loaded."

    try:
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )

        search_terms = optimize_query(user_query)

        # Better than "the and is..." : use a generic but meaningful fallback query
        if any(x in user_query.lower() for x in ["summary", "summarize", "overview"]):
            print("    ℹ️ Summary intent detected -> fetching broad representative chunks.")
            docs = docsearch.similarity_search("abstract conclusion key findings methodology", k=5)
        else:
            docs = docsearch.similarity_search(search_terms, k=4)

        if not docs:
            return "No relevant text found in the document (vector DB returned empty)."

        return "\n\n".join([d.page_content for d in docs if d.page_content])

    except Exception as e:
        return f"Error searching vector DB: {e}"

def process_files(files):
    """
    Extract text from uploaded files and ingest into Pinecone.

    Strategy:
      1) text/plain -> read directly
      2) PDF -> try pypdf page-by-page
      3) If pypdf yields no usable text, fall back to pdfminer
      4) Only report OCR required if BOTH extractors fail

    Returns:
      (ok: bool, details: str)
    """
    if not files:
        return False, "No files received."

    extracted_blocks = []
    total_pages = 0
    pages_with_text = 0
    used_pdfminer = False

    for file in files:
        # -------- TEXT FILES --------
        if file.type == "text/plain":
            try:
                with open(file.path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = (f.read() or "").strip()
                    if txt:
                        extracted_blocks.append(txt)
            except Exception as e:
                return False, f"Failed to read text file: {e}"

        # -------- PDF FILES --------
        elif (
            (file.type and "pdf" in file.type.lower())
            or str(file.name).lower().endswith(".pdf")
            ):

            # ---- Pass 1: pypdf ----
            try:
                reader = pypdf.PdfReader(file.path)

                # Encrypted PDF handling
                if getattr(reader, "is_encrypted", False):
                    try:
                        reader.decrypt("")  # empty password attempt
                    except Exception:
                        return False, "PDF is encrypted and cannot be read."

                for page in reader.pages:
                    total_pages += 1
                    try:
                        t = page.extract_text()
                    except Exception:
                        t = None

                    if t:
                        t = t.strip()
                        # Keep even small pages (title/abstract), but ignore tiny noise
                        if len(t) >= 10:
                            pages_with_text += 1
                            extracted_blocks.append(t)

            except Exception as e:
                # Don’t fail yet, we’ll try pdfminer
                print(f"⚠️ pypdf failed, trying pdfminer. Error: {e}")

            # ---- Pass 2: pdfminer fallback ----
            # Only if pypdf gave nothing useful for this PDF
            if pages_with_text == 0:
                try:
                    used_pdfminer = True
                    miner_text = pdfminer_extract_text(file.path)
                    if miner_text:
                        miner_text = miner_text.strip()
                        # pdfminer often includes whitespace; require some substance
                        if len(miner_text) >= 100:
                            extracted_blocks.append(miner_text)
                except Exception as e:
                    print(f"⚠️ pdfminer failed: {e}")

        else:
            # Ignore unsupported types, but tell user
            return False, f"Unsupported file type: {file.type}"

    # -------- FINAL VALIDATION --------
    full_text = "\n\n".join(extracted_blocks).strip()

    # HARD FAIL: no real text extracted
    if not full_text or len(full_text) < 100:
        return False, "Text extraction failed using pypdf and pdfminer (OCR required)."

    # -------- INGEST TO PINECONE --------
    try:
        ingest_to_pinecone(full_text)
    except Exception as e:
        return False, f"Pinecone ingest failed: {e}"

    extractor_used = "pdfminer" if used_pdfminer and pages_with_text == 0 else "pypdf"
    details = (
        f"Indexed successfully. "
        f"Extractor={extractor_used}, "
        f"pages={total_pages}, "
        f"pages_with_text={pages_with_text}, "
        f"chars={len(full_text)}"
    )
    return True, details


def is_file_related_question(question: str) -> bool:
    q = question.lower()
    file_keywords = [
        "this file", "this pdf", "this paper", "this document",
        "summarize", "summary", "explain this", "analyze this",
        "what is in this", "according to the paper", "according to the document"
    ]
    return any(k in q for k in file_keywords)


def get_next_step(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None

# --- 4. STARTUP ---
@cl.on_chat_start
async def start():
    # Reset session state for a fresh chat
    cl.user_session.set("graph", app)
    cl.user_session.set("memory", [])
    cl.user_session.set("use_pinecone", False)

    # Message shown ONLY when user clicks "New Chat"
    await cl.Message(
        content=(
            "🆕 **New chat started**\n\n"
            "ℹ️ Previous chats are saved as read-only history and can’t be continued.\n"
            "If you want to ask new questions or upload a new file, continue in this chat."
        ),
        author="System"
    ).send()

# --- 5. MAIN LOOP ---
@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("graph")
    memory = cl.user_session.get("memory")
    use_pinecone = cl.user_session.get("use_pinecone")

    # =========================
    # A) FILE UPLOAD HANDLING
    # =========================
    for f in message.elements:
        print("UPLOAD DEBUG:", f.name, f.type, getattr(f, "path", None))
    if message.elements:
        msg = cl.Message(content="⚙️ Processing file and indexing to Pinecone...")
        await msg.send()

        ok, details = process_files(message.elements)

        # Detect "file intent"
        file_intent = any(x in message.content.lower() for x in [
            "summarize", "summary", "this file", "this pdf", "this document",
            "explain this", "what is in this", "analyze this"
        ])

        if ok:
            use_pinecone = True
            cl.user_session.set("use_pinecone", True)
            msg.content = (
                "✅ **File Indexed!** Switched to **Vector Search Mode**.\n\n"
                f"({details})"
            )
            await msg.update()

        else:
            # ❗ HARD STOP FOR SCANNED PDFs
            cl.user_session.set("use_pinecone", False)
            use_pinecone = False

            msg.content = (
                "✅ **File Indexed!** Switched to **Vector Search Mode**.\n\n"
                f"**Details:** {details}"
                if ok else
                f"⚠️ **Indexing failed.**\n\nReason: {details}"
                )
            await msg.update()

            if file_intent:
                await cl.Message(
                    content=(
                        "I can’t summarize this PDF because it appears to be a "
                        "scanned/image-based file and no text could be extracted.\n\n"
                        "✅ What you can do:\n"
                        "1) Upload a text-based PDF (selectable text)\n"
                        "2) Run OCR on the PDF and re-upload\n\n"
                        "I will not guess or use web data for this file."
                    )
                ).send()
                return  # ⛔ ABSOLUTE STOP → prevents hallucination

    # =========================
    # B) PREPARE TASK
    # =========================
    file_related = is_file_related_question(message.content)

    if use_pinecone and file_related:
        # Use Pinecone ONLY for file-related questions
        vector_data = vector_search(message.content)
        full_task = (
            f"User Question: {message.content}\n\n"
            f"RELEVANT CONTEXT FROM PINECONE:\n{vector_data}"
            )
    else:
        # Use Web Agents
        context_string = "\n".join(
        [f"User: {m['user']}\nAI: {m['ai']}" for m in memory[-2:]]
        )
        full_task = (
            f"Context:\n{context_string}\n\nCurrent Question: {message.content}"
            if context_string else message.content
        )


    state = {
        "task": full_task,
        "research_data": [],
        "critique_feedback": "",
        "status": "starting",
        "revision_number": 0
    }

    msg = cl.Message(content="")
    await msg.send()
    
    mode = "📄 Document Mode (Pinecone)" if (use_pinecone and file_related) else "🌐 Web Research Mode"
    await cl.Message(content=f"**Mode:** {mode}").send()
                                                                                                                                                                                                                                                                  
    final_answer = ""

    # =========================
    # C) AGENT LOOP
    # =========================
    async with cl.Step(name="Agent Loop", type="run") as loop_step:
        loop_step.input = message.content
        iterator = graph.stream(state)

        while True:
            try:
                event = await cl.make_async(get_next_step)(iterator)
                if event is None:
                    break

                agent_name = list(event.keys())[0]
                data = event[agent_name]

                async with cl.Step(name=agent_name.title(), type="tool") as step:
                    if agent_name == "researcher":
                        chunk = (data.get("research_data") or [""])[-1]
                        step.output = (
                            "🧠 Retrieved vector context."
                            if use_pinecone
                            else f"🔎 Web search output size: {len(chunk)} chars"
                        )

                    elif agent_name == "critic":
                        status = data.get("status", "unknown")
                        feedback = data.get("critique_feedback", "")
                        step.output = f"🧐 Status: {status.upper()}\n{feedback}"

                    elif agent_name == "writer":
                        final_answer = data.get(
                            "final_report",
                            "I couldn’t generate a reliable answer."
                        )
                        step.output = "✍️ Final response generated."
                        msg.content = final_answer

            except Exception as e:
                print(f"CRITICAL ERROR: {e}")
                await cl.Message(
                    content=f"⛔ Process stopped.\nError: {e}"
                ).send()
                break

    if final_answer:
        memory.append({
            "user": message.content,
            "ai": final_answer[:200] + "..."
        })
        cl.user_session.set("memory", memory)

    await msg.update()