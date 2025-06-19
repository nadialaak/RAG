
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os
import shutil
import numpy as np
from langdetect import detect, DetectorFactory
import logging
# Fix for langdetect
DetectorFactory.seed = 0

# Import Hugging Face Transformers for the classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Configuration ---
CHROMA_PATH = "chroma_db"
EMBED_MODEL = "BAAI/bge-m3"
DOCUMENT_PATHS = {
    "lois": "data/lois",
    "depliants": "data/depliants",
    "actualites": "data/actualites"
}
LLMS_BY_DOMAIN = {
    "lois": "mistral",
    "depliants": "mistral",
    "actualites": "llama3"
}

# --- Classifier Model Configuration ---
CLASSIFIER_MODEL_PATH = "classifier_model"
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.75
custom_id_to_domain_mapping = {
    0: "lois",
    1: "depliants",
    2: "actualites"
}

# --- Global Variables ---
global_embedding_function = None
classifier_tokenizer = None
classifier_model = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def set_embedding_function(embedding_function):
    """Sets the global embedding function."""
    global global_embedding_function
    global_embedding_function = embedding_function
    logger.debug("Embedding function set in rag.py")

def load_classifier_model():
    """Loads the pre-trained classifier model and its tokenizer."""
    global classifier_tokenizer, classifier_model
    logger.debug(f"Loading classifier model from: {CLASSIFIER_MODEL_PATH}")
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        logger.warning(f"Classifier model path {CLASSIFIER_MODEL_PATH} does not exist. Skipping classifier loading.")
        return
    try:
        classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
        classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
        classifier_model.eval()
        logger.info("Classifier model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading classifier model: {e}")
        classifier_tokenizer = None
        classifier_model = None

def predict_domain_with_classifier(question: str):
    """Predicts the domain using the classifier model."""
    if classifier_tokenizer is None or classifier_model is None:
        logger.debug("Classifier not loaded")
        return None, 0.0
    try:
        inputs = classifier_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = classifier_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_id = torch.max(probabilities, dim=1)
        predicted_domain = custom_id_to_domain_mapping.get(predicted_id.item(), "unknown")
        logger.debug(f"Classifier predicted domain: {predicted_domain}, confidence: {confidence.item():.2f}")
        return predicted_domain, confidence.item()
    except Exception as e:
        logger.error(f"Classifier prediction failed: {e}")
        return None, 0.0

# --- Vector Store Management ---
def reset_chroma():
    """Removes existing ChromaDB directories."""
    for domain in DOCUMENT_PATHS:
        domain_path = CHROMA_PATH + f"_{domain}"
        if os.path.exists(domain_path):
            shutil.rmtree(domain_path)
            logger.info(f"ChromaDB for '{domain}' reset")
    logger.info("All ChromaDB instances reset")

def build_vectorstore(domain: str):
    """Builds a Chroma vector store for a domain."""
    docs = []
    path = DOCUMENT_PATHS[domain]
    logger.debug(f"Loading documents for domain: {domain} from {path}")
    if domain in ["lois", "depliants"]:
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, file))
                docs.extend(loader.load())
    elif domain == "actualites":
        for file in os.listdir(path):
            if file.endswith(".json"):
                loader = JSONLoader(
                    file_path=os.path.join(path, file),
                    jq_schema=".[] | {page_content: .resume, metadata: {source: .titre}}",
                    text_content=False
                )
                docs.extend(loader.load())
    else:
        raise ValueError(f"Unknown domain: {domain}")

    logger.debug(f"Found {len(docs)} documents for '{domain}'. Splitting into chunks")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)  # Reduced chunk_size
    chunks = splitter.split_documents(docs)
    logger.debug(f"Created {len(chunks)} chunks for '{domain}'")

    if global_embedding_function is None:
        logger.error("Global embedding function not set")
        raise Exception("Embedding function not initialized")
    logger.debug(f"Building vector store for '{domain}'")
    vs = Chroma.from_documents(chunks, embedding=global_embedding_function, persist_directory=CHROMA_PATH + f"_{domain}")
    logger.info(f"Vector store for '{domain}' built at {CHROMA_PATH}_{domain}")
    return vs

# --- Language Detection and Routing Labels ---
def detect_language(text):
    """Detects the language of the text."""
    try:
        return "ar" if detect(text) == "ar" else "fr"
    except:
        logger.warning("Language detection failed, defaulting to French")
        return "fr"

def translate_routing_label(label: str, target_lang: str) -> str:
    """Translates routing labels to the target language."""
    translations = {
        "fr": {
            "lois": "Ce document contient des textes de loi, décrets et règlements officiels. Il concerne les cadres légaux et les articles juridiques.",
            "depliants": "Ce document est un guide pratique, un formulaire ou une brochure d'information destinée aux usagers, décrivant des procédures, des pièces à fournir, ou des étapes à suivre.",
            "actualites": "Ce document contient des actualités, communiqués de presse ou informations récentes sur l'administration ou les services."
        },
        "ar": {
            "lois": "هذا المستند يتضمن نصوص القوانين والمراسيم واللوائح الرسمية. يتعلق بالأطر القانونية والمواد التشريعية.",
            "depliants": "هذا المستند هو دليل عملي، أو نموذج، أو كتيب إرشادي موجه للمستخدمين، يصف الإجراءات، الوثائق المطلوبة، أو الخطوات الواجب اتباعها.",
            "actualites": "هذا المستند يحتوي على أخبار، بلاغات صحفية، أو معلومات حديثة حول الإدارة أو الخدمات."
        }
    }
    try:
        return translations[target_lang][label]
    except KeyError:
        logger.warning(f"No translation for label '{label}' in '{target_lang}'. Using French default")
        return translations["fr"].get(label, "")

def cosine_similarity(vec1, vec2):
    """Calculates cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def route_question(question: str) -> list:
    """Routes the question to the most relevant domain."""
    logger.debug(f"Routing question: {question}")
    predicted_domain, confidence = predict_domain_with_classifier(question)

    if predicted_domain and predicted_domain in DOCUMENT_PATHS and confidence >= CLASSIFIER_CONFIDENCE_THRESHOLD:
        logger.info(f"Classifier predicted '{predicted_domain}' with confidence {confidence:.2f}")
        remaining_domains = [d for d in DOCUMENT_PATHS if d != predicted_domain]
        return [predicted_domain] + route_question_cosine_similarity(question, remaining_domains)
    else:
        if predicted_domain and predicted_domain not in DOCUMENT_PATHS:
            logger.warning(f"Classifier predicted unknown domain '{predicted_domain}'")
        elif classifier_tokenizer is None or classifier_model is None:
            logger.debug("Classifier not loaded, using cosine similarity")
        else:
            logger.debug(f"Classifier confidence too low ({confidence:.2f} < {CLASSIFIER_CONFIDENCE_THRESHOLD})")
        return route_question_cosine_similarity(question, list(DOCUMENT_PATHS.keys()))

def route_question_cosine_similarity(question: str, domains_to_consider: list) -> list:
    """Routes using cosine similarity."""
    logger.debug("Routing with cosine similarity")
    if global_embedding_function is None:
        logger.error("Global embedding function not set")
        raise Exception("Embedding function not initialized")
    question_lang = detect_language(question)
    labels_in_query_lang = {domain: translate_routing_label(domain, question_lang) for domain in domains_to_consider}
    q_emb = global_embedding_function.embed_query("query: " + question)
    scores = {
        k: cosine_similarity(q_emb, global_embedding_function.embed_query("query: " + v))
        for k, v in labels_in_query_lang.items()
    }
    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"Cosine similarity scores: {scores}")
    return [k for k, _ in sorted_domains]

# --- Prompt Management ---
def get_prompt(lang: str):
    """Returns the prompt template for the language."""
    if lang == "ar":
        return PromptTemplate.from_template(
"""أنت مساعد افتراضي متخصص في تحليل الوثائق باللغة العربية فقط.
➤ مهمتك هي استخراج إجابة دقيقة، واضحة، ومباشرة وفقًا للسؤال المطروح،
بالاعتماد فقط على المعلومات الواردة صراحة في الوثائق المقدمة.
➤ يجب أن تكون الإجابة:
- باللغة العربية فقط،
- خالية تمامًا من أي ترجمة أو شرح بلغة أخرى،
- دون أي تفسير أو إعادة صياغة تفقد المعنى الأصلي،
- بدون أي إضافات خارجية أو استنتاجات شخصية.
➤ لا تكتب أي تمهيد، شرح، ملاحظات، أو استنتاجات خارج نطاق الوثائق.
➤ إذا لم تكن المعلومة موجودة بوضوح في الوثائق، اكتب حرفيًا:
"المعلومة غير موجودة في الوثائق المتوفرة."
السؤال: {question}
الوثائق المرجعية: {context}
الإجابة:
"""
)
    return PromptTemplate.from_template(
"""Vous êtes un assistant virtuel spécialisé dans l’analyse documentaire.
Votre mission est de fournir une réponse exacte, complète et fidèle, exclusivement basée sur le contenu des documents suivants.
➤ Vous devez :
- Utiliser uniquement les informations **explicitement présentes** dans les documents,
- Reproduire **les formulations importantes** si nécessaire,
- Inclure **tous les éléments pertinents** sans les résumer ni les reformuler excessivement.
➤ Vous ne devez pas :
- Ajouter des informations extérieures ou des interprétations personnelles,
- Simplifier ou généraliser le contenu au point de perdre des détails essentiels.
Si la réponse ne se trouve pas dans les documents, répondez exactement :
"Information non trouvée dans les documents fournis."
Question : {question}
Documents de référence : {context}
Réponse :
"""
)

# --- RAG Chain Construction ---
def build_rag_chain(domain: str, question: str):
    """Builds the RAG chain for a domain and question."""
    logger.debug(f"Building RAG chain for domain: {domain}")
    lang = detect_language(question)
    prompt = get_prompt(lang)
    model_name = LLMS_BY_DOMAIN[domain]
    persist_path = CHROMA_PATH + f"_{domain}"
    if not os.path.exists(persist_path):
        logger.error(f"ChromaDB for '{domain}' not found at '{persist_path}'")
        raise ValueError(f"ChromaDB for domain '{domain}' not found")

    if global_embedding_function is None:
        logger.error("Global embedding function not set")
        raise Exception("Embedding function not initialized")

    retriever = Chroma(
        persist_directory=persist_path,
        embedding_function=global_embedding_function
    ).as_retriever(search_kwargs={"k": 1})  # k=1 pour limiter la recherche

    llm = OllamaLLM(model=model_name, temperature=0, timeout=20)  # paramètres du LLM

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    logger.debug(f"RAG chain built for domain: {domain}")
    return chain

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    load_classifier_model()
    if "--reset" in sys.argv:
        reset_chroma()
        for domain in DOCUMENT_PATHS:
            logger.info(f"Indexing {domain}")
            build_vectorstore(domain)
        logger.info("All vector stores built")
    logger.info("Multilingual RAG System Ready")
