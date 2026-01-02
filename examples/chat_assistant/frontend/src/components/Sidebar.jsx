import React from 'react'

function Sidebar({ conversations, currentConversation, onSelect, onNew, onDelete, isOpen, onToggle }) {
  const formatDate = (dateStr) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now - date
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  if (!isOpen) {
    return null
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Curio Chat</h2>
        <button className="new-chat-btn" onClick={onNew}>
          + New Chat
        </button>
      </div>

      <div className="conversations-list">
        {conversations.length === 0 ? (
          <div className="no-conversations">
            <p>No conversations yet</p>
            <p className="hint">Start a new chat to begin!</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${currentConversation?.id === conv.id ? 'active' : ''}`}
              onClick={() => onSelect(conv)}
            >
              <div className="conversation-title">{conv.title}</div>
              <div className="conversation-meta">
                <span className="conversation-date">{formatDate(conv.updated_at)}</span>
                {conv.message_count > 0 && (
                  <span className="conversation-count">{conv.message_count} msgs</span>
                )}
              </div>
              <button
                className="delete-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm('Delete this conversation?')) {
                    onDelete(conv.id)
                  }
                }}
                title="Delete conversation"
              >
                x
              </button>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <div className="sdk-info">
          <span>Powered by</span>
          <strong>Curio Agent SDK</strong>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
