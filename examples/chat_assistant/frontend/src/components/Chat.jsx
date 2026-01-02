import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function Chat({ messages, onSendMessage, isLoading, conversationTitle, onToggleSidebar, sidebarOpen }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <button className="menu-btn" onClick={onToggleSidebar} title="Toggle sidebar">
          {sidebarOpen ? '<<' : '>>'}
        </button>
        <h1>{conversationTitle}</h1>
        <div className="header-badge">Curio SDK Demo</div>
      </header>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <div className="welcome-icon">C</div>
            <h2>Welcome to Curio Chat Assistant</h2>
            <p>I'm a chat assistant powered by the Curio Agent SDK.</p>
            <p>I can help you with:</p>
            <ul>
              <li><strong>Math calculations</strong> - "What's 15% of 250?"</li>
              <li><strong>Unit conversions</strong> - "Convert 100 km to miles"</li>
              <li><strong>Advanced math</strong> - "What's the square root of 144?"</li>
              <li><strong>General chat</strong> - Ask me anything!</li>
            </ul>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? 'U' : 'C'}
            </div>
            <div className="message-content">
              <div className="message-role">
                {msg.role === 'user' ? 'You' : 'Curio'}
              </div>
              <div className={`message-text ${msg.error ? 'error' : ''}`}>
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.metadata && msg.metadata.tools_used && msg.metadata.tools_used.length > 0 && (
                <div className="message-metadata">
                  <span className="metadata-label">Tools used:</span>
                  {msg.metadata.tools_used.map((tool, i) => (
                    <span key={i} className="tool-badge">
                      {tool.action}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message assistant loading">
            <div className="message-avatar">C</div>
            <div className="message-content">
              <div className="message-role">Curio</div>
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="input-container" onSubmit={handleSubmit}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
          rows={1}
          disabled={isLoading}
        />
        <button type="submit" disabled={!input.trim() || isLoading}>
          {isLoading ? '...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default Chat
