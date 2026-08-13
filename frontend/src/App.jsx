import { useState } from 'react' // React hook for local component state
import Header from './components/Header.jsx' // top bar component
import MessageList from './components/MessageList.jsx' // renders the chat message bubbles
import ChatInput from './components/ChatInput.jsx' // text box + send button
import ReportPanel from './components/ReportPanel.jsx' // side panel showing the generated report
import { sendChatMessage } from './api.js' // calls the Flask REST API's /api/chat endpoint

const GREETING = "👋 Welcome! Let me help you research a niche.\n\nPlease enter a niche or category — e.g. **dog grooming**, **exercise band**, **home & kitchen**." // first assistant message shown on load

export default function App() { // root component for the whole app
  const [messages, setMessages] = useState([{ role: 'assistant', content: GREETING }]) // full chat history, seeded with the greeting
  const [isLoading, setIsLoading] = useState(false) // true while waiting on the backend reply
  const [reportPanel, setReportPanel] = useState(null) // currently displayed report, or null if none

  const hasStarted = messages.some((m) => m.role === 'user') // true once the user has sent at least one message
  const hasPanel = reportPanel !== null // true when a report is available to show

  const handleSend = async (text) => { // called when the user submits a message
    const updated = [...messages, { role: 'user', content: text }] // append the new user message to history
    setMessages(updated) // update state so the UI shows the user's message immediately
    setIsLoading(true) // show loading indicator while waiting for the API

    const { reply, ok } = await sendChatMessage(updated) // send full history to the backend and await the reply

    setMessages((m) => [...m, { role: 'assistant', content: reply }]) // append the assistant's reply to history
    if (ok) { // only populate the report panel on a successful response
      setReportPanel({ title: `📊 Report: ${text}`, content: reply }) // store the reply as a report keyed by the query text
    }
    setIsLoading(false) // hide loading indicator
  }

  return ( // render the app layout
    <div className={`app ${hasStarted ? 'active' : 'empty'}`}> {/* toggles layout class based on whether chat has started */}
      <Header /> {/* top bar */}
      <div className="main"> {/* main content area holding chat and report panel */}
        <div className="chat-column"> {/* left column: chat messages and input */}
          <MessageList messages={messages} isLoading={isLoading} /> {/* render message history and loading state */}
          <ChatInput onSend={handleSend} disabled={isLoading} /> {/* input box, disabled while loading */}
        </div>
        {hasPanel && ( // only render the report panel if a report exists
          <ReportPanel panel={reportPanel} onClear={() => setReportPanel(null)} /> // show report, allow clearing it
        )}
      </div>
    </div>
  )
}
