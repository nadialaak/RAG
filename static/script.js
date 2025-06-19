document.addEventListener('DOMContentLoaded', () => {
    // DOM elements
    const newConversationBtn = document.getElementById('new-conversation');
    const conversationList = document.getElementById('conversation-list');
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingDiv = document.getElementById('loading');
    let currentConversationId = null;
    let isSending = false;

    // Log function for debugging
    function log(message) {
        console.log(`[Chat] ${message}`);
    }

    // Check if DOM elements exist
    if (!conversationList || !chatMessages || !newConversationBtn || !userInput || !sendButton || !loadingDiv) {
        log('Error: One or more required DOM elements not found');
        return;
    }

    // Create a timeout-enabled fetch
    function fetchWithTimeout(url, options, timeout = 120000) {
        return new Promise((resolve, reject) => {
            const controller = new AbortController();
            const id = setTimeout(() => {
                controller.abort();
                reject(new Error('Request timed out'));
            }, timeout);
            fetch(url, { ...options, signal: controller.signal })
                .then(response => {
                    clearTimeout(id);
                    resolve(response);
                })
                .catch(error => {
                    clearTimeout(id);
                    reject(error);
                });
        });
    }

    // Handle unauthorized responses
    function handleUnauthorized() {
        log('User is unauthorized, redirecting to login');
        chatMessages.innerHTML += '<p class="text-red-500">Session expirée. Redirection vers la connexion...</p>';
        setTimeout(() => window.location.href = '/auth', 2000);
    }

    // Load conversations
    async function loadConversations(selectId = null) {
        log('Loading conversations');
        try {
            const response = await fetchWithTimeout('/api/conversations', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            });
            if (response.status === 401) {
                handleUnauthorized();
                return;
            }
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            const data = await response.json();
            conversationList.innerHTML = '';
            data.conversations.forEach(conv => {
                const convElement = document.createElement('div');
                convElement.className = `p-2 rounded cursor-pointer ${conv.id === (selectId || currentConversationId) ? 'bg-blue-100' : 'hover:bg-gray-100'}`;
                convElement.textContent = conv.titre || `Conversation ${conv.créé_le}`;
                convElement.addEventListener('click', () => loadMessages(conv.id));
                conversationList.appendChild(convElement);
            });
            log(`Loaded ${data.conversations.length} conversations`);
        } catch (error) {
            log(`Error loading conversations: ${error}`);
            chatMessages.innerHTML += '<p class="text-red-500">Erreur lors du chargement des conversations</p>';
        }
    }

    // Create new conversation
    newConversationBtn.addEventListener('click', async () => {
        log('Creating new conversation');
        try {
            const response = await fetchWithTimeout('/api/conversations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: `Conversation ${new Date().toLocaleString()}` }),
            });
            if (response.status === 401) {
                handleUnauthorized();
                return;
            }
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            const data = await response.json();
            currentConversationId = data.id_conversation;
            log(`New conversation created with ID: ${currentConversationId}`);
            await loadConversations(currentConversationId);
            await loadMessages(currentConversationId);
        } catch (error) {
            log(`Error creating conversation: ${error}`);
            chatMessages.innerHTML += '<p class="text-red-500">Erreur lors de la création de la conversation</p>';
        }
    });

    // Load messages for a conversation
    async function loadMessages(conversationId) {
        log(`Loading messages for conversation ID: ${conversationId}`);
        currentConversationId = conversationId;
        try {
            const response = await fetchWithTimeout(`/api/conversations/${conversationId}/messages`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
            });
            if (response.status === 401) {
                handleUnauthorized();
                return;
            }
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
            const data = await response.json();
            chatMessages.innerHTML = '';
            data.messages.forEach(msg => {
                addMessage('User', msg.message_utilisateur);
                addMessage('Bot', msg.réponse_bot);
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
            await loadConversations(conversationId);
            log(`Loaded ${data.messages.length} messages`);
        } catch (error) {
            log(`Error loading messages: ${error}`);
            chatMessages.innerHTML += '<p class="text-red-500">Erreur lors du chargement des messages</p>';
        }
    }

    // Add message to chat
    function addMessage(sender, text) {
        if (!text) return;
        const msgElement = document.createElement('div');
        msgElement.className = `p-2 mb-2 rounded ${sender === 'User' ? 'bg-blue-100 ml-auto max-w-[70%]' : 'bg-gray-100 mr-auto max-w-[70%]'}`;
        msgElement.innerHTML = `<strong>${sender}:</strong> ${text.replace(/\n/g, '<br>')}`;
        chatMessages.appendChild(msgElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Send message
    async function sendMessage() {
        if (isSending) {
            log('Request already in progress, ignoring');
            return;
        }
        const question = userInput.value.trim();
        log(`Sending message: ${question}, conversationId: ${currentConversationId}`);
        if (!question || !currentConversationId) {
            chatMessages.innerHTML += '<p class="text-red-500">Veuillez entrer une question et sélectionner une conversation</p>';
            log('Error: Missing question or conversation ID');
            return;
        }

        addMessage('User', question);
        userInput.value = '';
        isSending = true;
        sendButton.disabled = true;
        loadingDiv.classList.remove('hidden');

        let attempts = 0;
        const maxAttempts = 3;
        while (attempts < maxAttempts) {
            try {
                log('Starting non-streaming request');
                const response = await fetchWithTimeout('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, conversation_id: currentConversationId }),
                });
                const data = await response.json();
                if (response.status === 401) {
                    handleUnauthorized();
                    return;
                }
                if (!response.ok || data.status === 'error') throw new Error(data.response || `HTTP error: ${response.status}`);
                addMessage('Bot', data.response);
                log(`Non-streaming response received: ${data.response.substring(0, 100)}...`);
                return;
            } catch (error) {
                attempts++;
                log(`Attempt ${attempts} failed: ${error}`);
                if (attempts === maxAttempts) {
                    addMessage('Bot', `Erreur : Le serveur met trop de temps à répondre. Veuillez réessayer plus tard ou vérifier la configuration du modèle linguistique.`);
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            } finally {
                isSending = false;
                sendButton.disabled = false;
                loadingDiv.classList.add('hidden');
            }
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Initialize
    log('Initializing chat interface');
    loadConversations();
});