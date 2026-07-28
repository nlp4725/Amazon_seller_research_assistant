import { useState } from 'react'
import Header from './components/Header.jsx'
import MessageList from './components/MessageList.jsx'
import ChatInput from './components/ChatInput.jsx'
import ReportPanel from './components/ReportPanel.jsx'
import { sendChatMessage } from './api.js'

const GREETING = "👋 Welcome! Let me help you research a niche.\n\nPlease enter a niche or category — e.g. **dog grooming**, **exercise band**, **home & kitchen**."

export default function App() {
  const [messages, setMessages] = useState([{ role: 'assistant', content: GREETING }])
  const [isLoading, setIsLoading] = useState(false)
  const [reportPanel, setReportPanel] = useState(null)

  const hasStarted = messages.some((m) => m.role === 'user')
  const hasPanel = reportPanel !== null

  const handleSend = async (text) => {
    const updated = [...messages, { role: 'user', content: text }]
    setMessages(updated)
    setIsLoading(true)

    const { reply, ok } = await sendChatMessage(updated)

    setMessages((m) => [...m, { role: 'assistant', content: reply }])
    if (ok) {
      setReportPanel({ title: `📊 Report: ${text}`, content: reply })
    }
    setIsLoading(false)
  }

  return (
    <div className={`app ${hasStarted ? 'active' : 'empty'}`}>
      <Header />
      <div className="main">
        <div className="chat-column">
          <MessageList messages={messages} isLoading={isLoading} />
          <ChatInput onSend={handleSend} disabled={isLoading} />
        </div>
        {hasPanel && (
          <ReportPanel panel={reportPanel} onClear={() => setReportPanel(null)} />
        )}
      </div>
    </div>
  )
}
