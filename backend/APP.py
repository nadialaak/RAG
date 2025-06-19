from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import hashlib
import jwt
import datetime
from functools import wraps
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre-cle-secrete-tres-forte'  # Changez cette clé !
CORS(app)

# Configuration de la base de données
DATABASE = 'chatbot.db'

def init_db():
    """Initialiser la base de données avec les tables nécessaires"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Table des utilisateurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            prenom TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table des conversations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            titre TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Table des messages
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            is_user BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Obtenir une connexion à la base de données"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    """Hasher un mot de passe"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed_password):
    """Vérifier un mot de passe"""
    return hash_password(password) == hashed_password

def generate_token(user_id):
    """Générer un token JWT"""
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

def token_required(f):
    """Décorateur pour vérifier le token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token manquant'}), 401
        
        try:
            # Supprimer "Bearer " du token
            if token.startswith('Bearer '):
                token = token[7:]
            
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user_id = data['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expiré'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token invalide'}), 401
        
        return f(current_user_id, *args, **kwargs)
    
    return decorated

# Routes d'authentification

@app.route('/api/register', methods=['POST'])
def register():
    """Inscription d'un nouvel utilisateur"""
    data = request.get_json()
    
    if not all(k in data for k in ('nom', 'prenom', 'email', 'password')):
        return jsonify({'message': 'Données manquantes'}), 400
    
    conn = get_db_connection()
    
    # Vérifier si l'email existe déjà
    existing_user = conn.execute(
        'SELECT id FROM users WHERE email = ?', (data['email'],)
    ).fetchone()
    
    if existing_user:
        conn.close()
        return jsonify({'message': 'Cet email est déjà utilisé'}), 400
    
    # Créer le nouvel utilisateur
    hashed_password = hash_password(data['password'])
    
    cursor = conn.execute(
        'INSERT INTO users (nom, prenom, email, password) VALUES (?, ?, ?, ?)',
        (data['nom'], data['prenom'], data['email'], hashed_password)
    )
    
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Utilisateur créé avec succès', 'user_id': user_id}), 201

@app.route('/api/login', methods=['POST'])
def login():
    """Connexion d'un utilisateur"""
    data = request.get_json()
    
    if not all(k in data for k in ('email', 'password')):
        return jsonify({'message': 'Email et mot de passe requis'}), 400
    
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE email = ?', (data['email'],)
    ).fetchone()
    conn.close()
    
    if not user or not verify_password(data['password'], user['password']):
        return jsonify({'message': 'Email ou mot de passe incorrect'}), 401
    
    token = generate_token(user['id'])
    
    return jsonify({
        'token': token,
        'user': {
            'id': user['id'],
            'nom': user['nom'],
            'prenom': user['prenom'],
            'email': user['email']
        }
    }), 200

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile(current_user_id):
    """Mettre à jour le profil utilisateur"""
    data = request.get_json()
    
    conn = get_db_connection()
    
    # Construire la requête de mise à jour
    update_fields = []
    update_values = []
    
    if 'nom' in data:
        update_fields.append('nom = ?')
        update_values.append(data['nom'])
    
    if 'prenom' in data:
        update_fields.append('prenom = ?')
        update_values.append(data['prenom'])
    
    if 'email' in data:
        # Vérifier si le nouvel email n'est pas déjà utilisé
        existing_user = conn.execute(
            'SELECT id FROM users WHERE email = ? AND id != ?', 
            (data['email'], current_user_id)
        ).fetchone()
        
        if existing_user:
            conn.close()
            return jsonify({'message': 'Cet email est déjà utilisé'}), 400
        
        update_fields.append('email = ?')
        update_values.append(data['email'])
    
    if 'password' in data and data['password']:
        update_fields.append('password = ?')
        update_values.append(hash_password(data['password']))
    
    if update_fields:
        update_values.append(current_user_id)
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
        conn.execute(query, update_values)
        conn.commit()
    
    # Récupérer les données mises à jour
    user = conn.execute('SELECT * FROM users WHERE id = ?', (current_user_id,)).fetchone()
    conn.close()
    
    return jsonify({
        'message': 'Profil mis à jour avec succès',
        'user': {
            'id': user['id'],
            'nom': user['nom'],
            'prenom': user['prenom'],
            'email': user['email']
        }
    }), 200

# Routes pour les conversations

@app.route('/api/conversations', methods=['GET'])
@token_required
def get_conversations(current_user_id):
    """Récupérer les conversations de l'utilisateur"""
    conn = get_db_connection()
    conversations = conn.execute(
        '''SELECT c.*, 
           (SELECT content FROM messages m WHERE m.conversation_id = c.id AND m.is_user = 1 
            ORDER BY m.created_at ASC LIMIT 1) as first_message
           FROM conversations c 
           WHERE c.user_id = ? 
           ORDER BY c.updated_at DESC''',
        (current_user_id,)
    ).fetchall()
    conn.close()
    
    return jsonify([{
        'id': conv['id'],
        'titre': conv['titre'],
        'created_at': conv['created_at'],
        'updated_at': conv['updated_at'],
        'first_message': conv['first_message']
    } for conv in conversations])

@app.route('/api/conversations', methods=['POST'])
@token_required
def create_conversation(current_user_id):
    """Créer une nouvelle conversation"""
    data = request.get_json()
    titre = data.get('titre', 'Nouvelle conversation')
    
    conn = get_db_connection()
    cursor = conn.execute(
        'INSERT INTO conversations (user_id, titre) VALUES (?, ?)',
        (current_user_id, titre)
    )
    
    conversation_id = cursor.lastrowid
    conn.commit()
    
    # Récupérer la conversation créée
    conversation = conn.execute(
        'SELECT * FROM conversations WHERE id = ?', (conversation_id,)
    ).fetchone()
    conn.close()
    
    return jsonify({
        'id': conversation['id'],
        'titre': conversation['titre'],
        'created_at': conversation['created_at'],
        'updated_at': conversation['updated_at']
    }), 201

@app.route('/api/conversations/<int:conversation_id>/messages', methods=['GET'])
@token_required
def get_messages(current_user_id, conversation_id):
    """Récupérer les messages d'une conversation"""
    conn = get_db_connection()
    
    # Vérifier que la conversation appartient à l'utilisateur
    conversation = conn.execute(
        'SELECT * FROM conversations WHERE id = ? AND user_id = ?',
        (conversation_id, current_user_id)
    ).fetchone()
    
    if not conversation:
        conn.close()
        return jsonify({'message': 'Conversation non trouvée'}), 404
    
    messages = conn.execute(
        'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
        (conversation_id,)
    ).fetchall()
    conn.close()
    
    return jsonify([{
        'id': msg['id'],
        'content': msg['content'],
        'is_user': bool(msg['is_user']),
        'created_at': msg['created_at']
    } for msg in messages])

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user_id):
    """Traiter un message du chat et retourner la réponse"""
    data = request.get_json()
    
    if 'message' not in data or 'conversation_id' not in data:
        return jsonify({'message': 'Message et conversation_id requis'}), 400
    
    user_message = data['message']
    conversation_id = data['conversation_id']
    
    conn = get_db_connection()
    
    # Vérifier que la conversation appartient à l'utilisateur
    conversation = conn.execute(
        'SELECT * FROM conversations WHERE id = ? AND user_id = ?',
        (conversation_id, current_user_id)
    ).fetchone()
    
    if not conversation:
        conn.close()
        return jsonify({'message': 'Conversation non trouvée'}), 404
    
    # Sauvegarder le message utilisateur
    conn.execute(
        'INSERT INTO messages (conversation_id, content, is_user) VALUES (?, ?, ?)',
        (conversation_id, user_message, True)
    )
    
    # ICI : Intégrer votre code RAG
    # Remplacez cette partie par votre logique RAG
    bot_response = process_with_rag(user_message)
    
    # Sauvegarder la réponse du bot
    conn.execute(
        'INSERT INTO messages (conversation_id, content, is_user) VALUES (?, ?, ?)',
        (conversation_id, bot_response, False)
    )
    
    # Mettre à jour la date de dernière modification de la conversation
    conn.execute(
        'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
        (conversation_id,)
    )
    
    conn.commit()
    conn.close()
    
    return jsonify({'response': bot_response})

def process_with_rag(user_message):
    """
    REMPLACEZ CETTE FONCTION PAR VOTRE CODE RAG
    
    Cette fonction doit :
    1. Prendre le message utilisateur en entrée
    2. Traiter le message avec votre système RAG
    3. Retourner la réponse générée
    
    Exemple d'intégration :
    - Appelez votre fonction de recherche dans la base de connaissances
    - Utilisez le contexte trouvé pour générer une réponse
    - Retournez la réponse finale
    """
    
    # EXEMPLE - Remplacez par votre code RAG
    try:
        # Ici vous devriez appeler votre système RAG
        # Par exemple :
        # context = search_knowledge_base(user_message)
        # response = generate_response(user_message, context)
        # return response
        
        # Pour l'instant, retour d'une réponse générique
        return f"Merci pour votre question : '{user_message}'. Je suis en cours de développement et j'intégrerai bientôt votre système RAG !"
        
    except Exception as e:
        print(f"Erreur dans le traitement RAG : {e}")
        return "Désolé, une erreur est survenue lors du traitement de votre question."

# Route de test
@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérifier l'état de l'API"""
    return jsonify({'status': 'OK', 'message': 'API ChatBot RAG fonctionnelle'})

if __name__ == '__main__':
    # Initialiser la base de données
    init_db()
    print("Base de données initialisée avec succès !")
    print("API ChatBot RAG démarrée sur http://localhost:5000")
    print("\nEndpoints disponibles :")
    print("- POST /api/register : Inscription")
    print("- POST /api/login : Connexion") 
    print("- PUT /api/profile : Mise à jour profil")
    print("- GET /api/conversations : Liste des conversations")
    print("- POST /api/conversations : Créer une conversation")
    print("- GET /api/conversations/<id>/messages : Messages d'une conversation")
    print("- POST /api/chat : Envoyer un message au chatbot")
    print("- GET /api/health : Test de l'API")
    
    app.run(debug=True, host='0.0.0.0', port=5000)