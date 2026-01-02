import React, { useState, useEffect } from 'react'
import Chat from './components/Chat'
import Sidebar from './components/Sidebar'

const API_BASE = '/api'

function App() {
  const [conversations, setConversations] = useState([])
  const [currentConversation, setCurrentConversation] = useState(null)
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Load conversations on mount
  useEffect(() => {
    loadConversations()
  }, [])

  // Load messages when conversation changes
  useEffect(() => {
    if (currentConversation) {
      loadMessages(currentConversation.id)
    } else {
      setMessages([])
    }
  }, [currentConversation])

  const loadConversations = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`)
      const data = await res.json()
      setConversations(data.conversations || [])
    } catch (err) {
      console.error('Failed to load conversations:', err)
    }
  }

  const loadMessages = async (conversationId) => {
    try {
      const res = await fetch(`${API_BASE}/conversations/${conversationId}`)
      const data = await res.json()
      setMessages(data.messages || [])
    } catch (err) {
      console.error('Failed to load messages:', err)
    }
  }

  const createNewConversation = async () => {
    setCurrentConversation(null)
    setMessages([])
  }

  const selectConversation = (conversation) => {
    setCurrentConversation(conversation)
  }

  const deleteConversation = async (conversationId) => {
    try {
      await fetch(`${API_BASE}/conversations/${conversationId}`, {
        method: 'DELETE'
      })
      await loadConversations()
      if (currentConversation?.id === conversationId) {
        setCurrentConversation(null)
        setMessages([])
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }

  const sendMessage = async (message) => {
    if (!message.trim()) return

    // Optimistically add user message
    const userMessage = {
      role: 'user',
      content: message,
      created_at: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: currentConversation?.id,
          message
        })
      })

      const data = await res.json()

      // Update conversation if new
      if (!currentConversation && data.conversation_id) {
        const newConv = {
          id: data.conversation_id,
          title: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
          created_at: new Date().toISOString()
        }
        setCurrentConversation(newConv)
        setConversations(prev => [newConv, ...prev])
      }

      // Add assistant response
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        metadata: data.metadata,
        created_at: new Date().toISOString()
      }
      setMessages(prev => [...prev, assistantMessage])

      // Refresh conversations list
      await loadConversations()
    } catch (err) {
      console.error('Failed to send message:', err)
      // Add error message
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
        error: true,
        created_at: new Date().toISOString()
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversation={currentConversation}
        onSelect={selectConversation}
        onNew={createNewConversation}
        onDelete={deleteConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      <main className={`main-content ${sidebarOpen ? '' : 'sidebar-closed'}`}>
        <Chat
          messages={messages}
          onSendMessage={sendMessage}
          isLoading={isLoading}
          conversationTitle={currentConversation?.title || 'New Conversation'}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          sidebarOpen={sidebarOpen}
        />
      </main>
    </div>
  )
}

export default App
