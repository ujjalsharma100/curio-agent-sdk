"""
Flask Backend for Chat Assistant

Provides REST API endpoints for the chat interface.
Manages conversation history in SQLite.
"""

import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from contextlib import contextmanager

from agent import create_agent

app = Flask(__name__)
CORS(app)

# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'chat_history.db')


def init_db():
    """Initialize the SQLite database with required tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        ''')

        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        ''')

        conn.commit()


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def generate_conversation_id():
    """Generate a unique conversation ID."""
    import uuid
    return str(uuid.uuid4())[:8]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all conversations."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''')
        conversations = [dict(row) for row in cursor.fetchall()]

    return jsonify({'conversations': conversations})


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """Create a new conversation."""
    conversation_id = generate_conversation_id()
    title = request.json.get('title', 'New Conversation')

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (id, title) VALUES (?, ?)',
            (conversation_id, title)
        )
        conn.commit()

    return jsonify({
        'id': conversation_id,
        'title': title,
        'created_at': datetime.utcnow().isoformat()
    })


@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get a conversation with all its messages."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get conversation details
        cursor.execute(
            'SELECT * FROM conversations WHERE id = ?',
            (conversation_id,)
        )
        conversation = cursor.fetchone()

        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404

        # Get messages
        cursor.execute(
            'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        messages = []
        for row in cursor.fetchall():
            msg = dict(row)
            if msg.get('metadata'):
                msg['metadata'] = json.loads(msg['metadata'])
            messages.append(msg)

    return jsonify({
        'conversation': dict(conversation),
        'messages': messages
    })


@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation and all its messages."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        conn.commit()

    return jsonify({'status': 'deleted'})


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Send a message and get a response from the agent.

    Request body:
    {
        "conversation_id": "optional-id",
        "message": "user message"
    }
    """
    data = request.json
    user_message = data.get('message', '').strip()
    conversation_id = data.get('conversation_id')

    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # Create new conversation if needed
    if not conversation_id:
        conversation_id = generate_conversation_id()
        # Use first few words as title
        title = user_message[:50] + ('...' if len(user_message) > 50 else '')
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO conversations (id, title) VALUES (?, ?)',
                (conversation_id, title)
            )
            conn.commit()

    # Get conversation history for context
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        history = [dict(row) for row in cursor.fetchall()]

    # Save user message
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)',
            (conversation_id, 'user', user_message)
        )
        cursor.execute(
            'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )
        conn.commit()

    # Add current message to history for agent context
    history.append({'role': 'user', 'content': user_message})

    try:
        # Create agent with conversation history
        agent = create_agent(conversation_history=history)

        # Run the agent
        result = agent.run(
            objective=user_message,
            additional_context={
                'conversation_id': conversation_id,
                'message_count': len(history)
            }
        )

        # Extract response
        assistant_response = result.synthesis_summary if result.status == 'done' else f"Error: {result.error}"

        # Build metadata about the agent run
        metadata = {
            'run_id': result.run_id,
            'iterations': result.total_iterations,
            'status': result.status,
            'tools_used': []
        }

        # Extract tools used from execution history
        for item in result.execution_history:
            if item.get('step') == 'action_executed':
                metadata['tools_used'].append({
                    'action': item.get('action'),
                    'result_status': item.get('observation', {}).get('status')
                })

        # Save assistant response
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)',
                (conversation_id, 'assistant', assistant_response, json.dumps(metadata))
            )
            conn.commit()

        return jsonify({
            'conversation_id': conversation_id,
            'response': assistant_response,
            'metadata': metadata
        })

    except Exception as e:
        error_msg = f"Agent error: {str(e)}"

        # Save error as assistant message
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO messages (conversation_id, role, content, metadata) VALUES (?, ?, ?, ?)',
                (conversation_id, 'assistant', error_msg, json.dumps({'error': True}))
            )
            conn.commit()

        return jsonify({
            'conversation_id': conversation_id,
            'response': error_msg,
            'error': True
        }), 500


@app.route('/api/conversations/<conversation_id>/messages', methods=['DELETE'])
def clear_messages(conversation_id):
    """Clear all messages in a conversation but keep the conversation."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
        cursor.execute(
            'UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )
        conn.commit()

    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    # Initialize database
    init_db()
    print("Database initialized at:", DB_PATH)

    # Run Flask app
    port = int(os.environ.get('PORT', 5004))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print(f"Starting Chat Assistant API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
