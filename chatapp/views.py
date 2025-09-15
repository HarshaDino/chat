import os
import uuid
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from pathlib import Path

from .models import Document
from .serializers import DocumentSerializer
from . import utils

# Globals to cache heavy objects so generation is faster after first request
_MODEL_PIPELINE = None
_EMBEDDER = None
_CHROMA_CLIENT = None
_CHROMA_COLL = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = utils.get_embedding_model()
    return _EMBEDDER

def get_chroma():
    global _CHROMA_CLIENT, _CHROMA_COLL
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT, _CHROMA_COLL = utils.get_chroma_client_and_collection()
    return _CHROMA_CLIENT, _CHROMA_COLL

def get_model_pipeline():
    """
    Lazy-load a very small text2text pipeline for speed.
    Default model: sshleifer/tiny-gpt2 (very small) — fast but limited quality.
    You can set MODEL_NAME env to change it.
    """
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        model_name = os.environ.get('MODEL_NAME', 'sshleifer/tiny-gpt2')
        try:
            from transformers import pipeline
            # device - CPU; pipeline will reuse resources between calls
            _MODEL_PIPELINE = pipeline('text-generation', model=model_name, tokenizer=model_name)
        except Exception as e:
            # keep None and handle later
            _MODEL_PIPELINE = None
    return _MODEL_PIPELINE

def index(request):
    return render(request, 'index.html', {})

class UploadView(APIView):
    def post(self, request, format=None):
        f = request.FILES.get('file')
        if not f:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        doc = Document.objects.create(file=f, original_name=getattr(f, 'name', None))
        doc_path = doc.file.path
        text = utils.extract_text_from_file(doc_path)
        chunks = utils.chunk_text(text)
        if chunks:
            embedder = get_embedder()
            embeddings = utils.embed_texts(chunks, embedder=embedder)
            client, coll = get_chroma()
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{'doc_id': doc.id, 'source': doc.original_name or doc.file.name, 'chunk_index': i} for i in range(len(chunks))]
            coll.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
            client.persist()
        serializer = DocumentSerializer(doc)
        return Response(serializer.data)

class ChatView(APIView):
    def post(self, request, format=None):
        """
        Request body: { "question": "...", "mode": "doc" | "general" }
        mode = "doc" -> use RAG (retrieve from stored docs)
        mode = "general" -> answer using the model only (no retrieval)
        If mode omitted, will try RAG and fall back to general answer.
        """
        question = request.data.get('question')
        mode = request.data.get('mode', '').lower()
        if not question:
            return Response({'error': 'Missing question'}, status=status.HTTP_400_BAD_REQUEST)

        embedder = get_embedder()
        model_pipe = get_model_pipeline()

        # Embed question
        try:
            q_emb = embedder.encode([question], show_progress_bar=False, convert_to_numpy=True).tolist()[0]
        except Exception as e:
            return Response({'error': f'Embedding error: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Try retrieval if mode != 'general'
        context = ''
        sources = []
        if mode != 'general':
            try:
                client, coll = get_chroma()
                result = coll.query(query_embeddings=[q_emb], n_results=3)
                docs = []
                metas = []
                for docs_list in result.get('documents', []):
                    docs.extend(docs_list)
                for metas_list in result.get('metadatas', []):
                    metas.extend(metas_list)
                if docs:
                    context = '\n\n---\n\n'.join(docs[:3])
                    sources = metas[:3]
            except Exception:
                # retrieval failed silently — we'll fall back to general generation
                context = ''

        # If no context found or user explicitly asked general -> general answer
        answer = generate_answer(question, context, model_pipe)
        return Response({'answer': answer, 'context': context, 'sources': sources})

def generate_answer(question: str, context: str, model_pipe) -> str:
    """
    Generate an answer using the cached model pipeline.
    - If model pipeline is not available, returns a helpful fallback message.
    - Limit tokens for speed.
    """
    # small instruction prompt
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    if model_pipe is None:
        return "Sorry — model unavailable on server. Try restarting the app or check logs."

    try:
        # Use generation with small max_new_tokens for speed
        out = model_pipe(prompt, max_new_tokens=80, do_sample=False, num_return_sequences=1)
        if isinstance(out, list) and len(out) > 0:
            text = out[0].get('generated_text') or out[0].get('text') or str(out[0])
            return text.strip()
        return str(out).strip()
    except Exception as e:
        return f"Model generation failed: {e}"


class UploadDocumentView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return Response({"error":"No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        # Basic validation
        if f.size > 20 * 1024 * 1024:
            return Response({"error":"File too large (max 20MB)"}, status=status.HTTP_400_BAD_REQUEST)
        fname = f"{uuid.uuid4().hex}_{f.name}"
        media_path = Path(settings.MEDIA_ROOT)
        media_path.mkdir(parents=True, exist_ok=True)
        dest = media_path / fname
        with open(dest, "wb") as out:
            for chunk in f.chunks():
                out.write(chunk)
        # extract text
        text_pages = extract_text_from_file(str(dest))
        if not text_pages:
            return Response({"error":"Could not extract text. Is this a scanned PDF?"}, status=status.HTTP_400_BAD_REQUEST)
        # add to vector store
        add_documents_to_vector_store(text_pages, doc_id=fname)
        return Response({"status":"ok","doc_id": fname})

# chatapp/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import get_top_k_context, generate_answer

@api_view(["POST"])
def chat(request):
    q = request.data.get("question")
    if not q:
        return Response({"error":"Missing question"}, status=400)
    top_context = get_top_k_context(q, k=4)  # returns list of strings
    answer = generate_answer(question=q, contexts=top_context)
    return Response({"answer": answer, "context": top_context})

# chatapp/utils.py continued
def get_top_k_context(query: str, k=4):
    col = client.get_collection("documents")
    # compute embedding via ef.embedding_function if needed
    results = col.query(query_texts=[query], n_results=k)
    # results['documents'] is list of lists
    docs = results["documents"][0] if "documents" in results else []
    return docs

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
MODEL_NAME = os.getenv("MODEL_NAME","google/flan-t5-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def generate_answer(question: str, contexts: list):
    # Build a prompt: short instructions + contexts + question
    ctx_text = "\n\n".join([f"Context {i+1}: {c}" for i,c in enumerate(contexts)])
    prompt = f"Use the following contexts to answer the question. If the answer is not in the contexts, say 'I don't know'.\n\n{ctx_text}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=2)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    return ans
