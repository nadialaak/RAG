import re
from flask import Flask, request, jsonify, render_template, session, flash, redirect, url_for
from flask_bcrypt import Bcrypt
import sqlite3
from datetime import datetime
from rag import route_question, build_rag_chain, set_embedding_function
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory
import os
import logging
import requests
import torch
import time
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Fix for langdetect
DetectorFactory.seed = 0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
bcrypt = Bcrypt(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize embedding function and spell checker
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": device})
set_embedding_function(embedding_function)  # Pass to rag.py
spell = SpellChecker(language='fr')
spell.word_frequency.load_words(['rcar'])  # Add 'rcar' to dictionary

# Database connection
def get_db():
    conn = sqlite3.connect('chatbot.db', timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

# Detect language
def detect_language(text):
    try:
        return "ar" if detect(text) == "ar" else "fr"
    except:
        logger.warning("Language detection failed, defaulting to French")
        return "fr"

# Routes
@app.route('/')
def index():
    logger.debug("Serving index page")
    return render_template('base.html')

@app.route('/auth')
def auth():
    logger.debug("Serving auth page")
    if session.get('user_id'):
        return redirect(url_for('chat_page'))
    return render_template('auth.html')

@app.route('/login', methods=['POST'])
def login():
    logger.debug("Processing login request")
    email = request.form['email']
    password = request.form['password']

    # Init attempts in session
    session.setdefault('login_attempts', 0)

    if session['login_attempts'] >= 5:
        flash("Trop de tentatives. Réessayez plus tard.", "danger")
        return redirect(url_for('auth'))

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.check_password_hash(user['password_hash'], password):
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        session['login_attempts'] = 0  # reset on success
        flash('Connexion réussie !', 'success')
        return redirect(url_for('chat_page'))
    else:
        session['login_attempts'] += 1
        flash('Identifiants incorrects', 'danger')
        return redirect(url_for('auth'))

@app.route('/register', methods=['POST'])
def register():
    logger.debug("Traitement de la demande d'inscription")

    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    email = request.form.get('email', '').strip()
    full_name = request.form.get('full_name', '').strip()

    # Vérifications de base
    if not username or not password or not email:
        flash("Tous les champs obligatoires doivent être remplis.", 'danger')
        return redirect(url_for('auth', mode='register'))

    if len(password) < 6:
        flash("Le mot de passe doit contenir au moins 6 caractères.", 'danger')
        return redirect(url_for('auth', mode='register'))

    # Vérification si username ou email déjà utilisés
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (username, email))
    if cursor.fetchone():
        flash("Ce nom d'utilisateur ou cet email est déjà utilisé.", 'danger')
        conn.close()
        return redirect(url_for('auth', mode='register'))

    # Création du compte
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, full_name) VALUES (?, ?, ?, ?)",
            (username, password_hash, email, full_name)
        )
        conn.commit()
        flash('Inscription réussie ! Veuillez vous connecter.', 'success')
        return redirect(url_for('auth'))
    except Exception as e:
        logger.error(f"Erreur lors de l'inscription : {e}")
        flash("Une erreur est survenue lors de l'inscription. Veuillez réessayer.", 'danger')
        return redirect(url_for('auth', mode='register'))
    finally:
        conn.close()


@app.route('/logout')
def logout():
    logger.debug("Déconnexion en cours")
    session.clear()
    flash('Déconnexion réussie.', 'success')
    return redirect(url_for('auth'))

@app.route('/chat')
def chat_page():
    logger.debug("Affichage de la page de discussion")
    if not session.get('user_id'):
        flash('Veuillez vous connecter pour accéder au chat.', 'danger')
        return redirect(url_for('auth'))
    return render_template('chat.html')

@app.route('/profile')
def profile_page():
    logger.debug("Affichage de la page de profil")
    if not session.get('user_id'):
        flash('Veuillez vous connecter pour accéder au profil.', 'danger')
        return redirect(url_for('auth'))
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    logger.debug("Mise à jour du profil utilisateur")
    if not session.get('user_id'):
        flash('Veuillez vous connecter pour modifier votre profil.', 'danger')
        return redirect(url_for('auth'))

    email = request.form.get('email')
    full_name = request.form.get('full_name')
    new_password = request.form.get('new_password')
    current_password = request.form.get('current_password')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE user_id = ?", (session['user_id'],))
    user = cursor.fetchone()

    if new_password and not bcrypt.check_password_hash(user['password_hash'], current_password):
        flash('Le mot de passe actuel est incorrect.', 'danger')
        conn.close()
        return redirect(url_for('profile_page'))

    update_query = "UPDATE users SET email = ?, full_name = ?"
    update_params = [email, full_name]

    if new_password:
        password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
        update_query += ", password_hash = ?"
        update_params.append(password_hash)

    update_query += " WHERE user_id = ?"
    update_params.append(session['user_id'])

    cursor.execute(update_query, update_params)
    conn.commit()
    conn.close()
    flash('Profil mis à jour avec succès.', 'success')
    return redirect(url_for('profile_page'))

@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    logger.debug(f"Requête POST reçue sur /api/conversations : en-têtes={request.headers}, corps={request.get_data(as_text=True)}")
    if not session.get('user_id'):
        return jsonify({'erreur': 'Non autorisé'}), 401

    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        return jsonify({'erreur': 'JSON invalide'}), 400

    title = data.get('title', f"Conversation du {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", (session['user_id'], title))
        conversation_id = cursor.lastrowid
        conn.commit()
        logger.debug(f"Conversation créée avec l'ID : {conversation_id}")
        return jsonify({'id_conversation': conversation_id, 'titre': title}), 201
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la création de la conversation : {e}")
        return jsonify({'erreur': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    logger.debug(f"Requête GET reçue sur /api/conversations : en-têtes={request.headers}")
    if not session.get('user_id'):
        return jsonify({'erreur': 'Non autorisé'}), 401
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT conversation_id, title, created_at FROM conversations WHERE user_id = ?",
        (session['user_id'],)
    )
    conversations = [{
        'id': row['conversation_id'],
        'titre': row['title'],
        'créé_le': row['created_at']
    } for row in cursor.fetchall()]
    conn.close()
    logger.debug(f"{len(conversations)} conversations retournées")
    return jsonify({'conversations': conversations}), 200


@app.route('/api/conversations/<int:conversation_id>/messages', methods=['GET'])
def get_messages(conversation_id):
    logger.debug(f"Requête GET reçue sur /api/conversations/{conversation_id}/messages : en-têtes={request.headers}")
    if not session.get('user_id'):
        return jsonify({'erreur': 'Non autorisé'}), 401
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_message, bot_response, timestamp FROM messages WHERE conversation_id = ?",
        (conversation_id,)
    )
    messages = [{
        'message_utilisateur': row['user_message'],
        'réponse_bot': row['bot_response'],
        'horodatage': row['timestamp']
    } for row in cursor.fetchall()]
    conn.close()
    logger.debug(f"{len(messages)} messages retournés pour la conversation {conversation_id}")
    return jsonify({'messages': messages}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    logger.debug(f"Requête POST reçue sur /api/chat : en-têtes={request.headers}, corps={request.get_data(as_text=True)}")
    
    if not session.get('user_id'):
        logger.error("Tentative d'accès non autorisée")
        return jsonify({'status': 'error', 'response': 'Non autorisé'}), 401

    data = request.get_json()
    question = data.get('question')
    conversation_id = data.get('conversation_id')

    if not question or not conversation_id:
        logger.error("Question ou identifiant de conversation manquant")
        return jsonify({'status': 'error', 'response': 'La question et l\'identifiant de la conversation sont requis.'}), 400

    # Détection de la langue arabe
    if re.search(r'[\u0600-\u06FF]', question):
        corrected_words = question.split()
    else:
        words = question.split()
        corrected_words = [spell.correction(word) if spell.unknown([word]) else word for word in words]

    corrected_question = ' '.join([word for word in corrected_words if word is not None])

    if corrected_question != question:
        logger.debug(f"Question corrigée : {question} -> {corrected_question}")

    try:
        logger.debug("Vérification de la disponibilité du serveur Ollama")
        response = requests.get('http://127.0.0.1:11434', timeout=5)
        if response.status_code != 200:
            raise Exception(f"Ollama a répondu avec le code {response.status_code}")
    except Exception as e:
        logger.error(f"Échec de la vérification Ollama : {e}")
        return jsonify({'status': 'error', 'response': 'Le serveur du modèle linguistique est indisponible.'}), 503

    logger.debug(f"Routage de la question : {corrected_question}")
    candidate_domains = route_question(corrected_question)
    logger.debug(f"Domaines candidats : {candidate_domains}")

    domain_found = None
    docs_for_domain = None

    for domain in candidate_domains:
        current_persist_path = f"chroma_db_{domain}"
        if not os.path.exists(current_persist_path):
            logger.warning(f"Base de données Chroma introuvable : {current_persist_path}")
            continue

        logger.debug(f"Chargement de la base de données Chroma : {current_persist_path}")
        current_retriever = Chroma(
            persist_directory=current_persist_path,
            embedding_function=embedding_function
        ).as_retriever(search_kwargs={"k": 1})

        docs_for_domain = current_retriever.invoke(corrected_question)
        logger.debug(f"{len(docs_for_domain)} documents récupérés pour le domaine {domain}")

        if docs_for_domain:
            domain_found = domain
            break

    if domain_found is None:
        logger.error("Aucun document pertinent trouvé pour les domaines analysés")
        return jsonify({'status': 'error', 'response': 'Aucun document pertinent trouvé.'}), 404

    logger.debug(f"Construction de la chaîne RAG pour le domaine : {domain_found}")
    start_time = time.time()
    logger.debug(">> Construction de la chaîne RAG")

    rag_chain = build_rag_chain(domain_found, corrected_question)

    import signal
    class TimeoutException(Exception): pass
    def timeout_handler(signum, frame): raise TimeoutException()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(25)

    try:
        logger.debug(">> Lancement de la génération via rag_chain.invoke()")
        result = rag_chain.invoke({"query": corrected_question})
        response = result["result"]
        signal.alarm(0)
    except TimeoutException:
        logger.error(" Délai dépassé : le modèle  a mis trop de temps à répondre.")
        return jsonify({'status': 'error', 'response': 'Le modèle a mis trop de temps à répondre.'}), 504

    logger.debug(f"Réponse du modèle  : {response[:100]}...")
    logger.debug(f"Durée de traitement : {time.time() - start_time} secondes")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (conversation_id, user_message, bot_response) VALUES (?, ?, ?)",
                   (conversation_id, question, response))
    conn.commit()
    conn.close()

    logger.debug("Envoi de la réponse au frontend")
    return jsonify({'status': 'success', 'response': response}), 200

if __name__ == "__main__":
    logger.info("Démarrage de l'application Flask")
    app.run(debug=True, port=5000, use_reloader=False)
