import { useEffect, useMemo, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const statusLabel = {
  supported: 'Supported',
  uncertain: 'Needs review',
  unsupported: 'Unsupported',
}

function createConversation() {
  return { id: crypto.randomUUID(), title: 'New conversation', messages: [] }
}

function VerificationCard({ result }) {
  if (!result?.claims?.length) return null
  const reliability = Math.round((result.reliability_score ?? 0) * 100)

  return <details className="verification-card">
    <summary><span className="verification-dot"></span>Verification available<strong>{reliability}% reliable</strong></summary>
    <div className="verification-content">
      <p className="verification-summary">{result.message}</p>
      <div className="claim-list">
        {result.claims.map((claim, index) => <article className="claim" key={`${claim.claim}-${index}`}>
          <span className={`claim-dot ${claim.status}`}></span>
          <div><b>{claim.claim}</b><p>{statusLabel[claim.status]} · {Math.round(claim.confidence * 100)}% confidence</p></div>
        </article>)}
      </div>
      {result.evidence?.length > 0 && <div className="citations"><span>Sources</span>
        {result.evidence.map((source, index) => source.url.startsWith('document://')
          ? <em key={`${source.url}-${index}`}>[{index + 1}] {source.title}</em>
          : <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">[{index + 1}] {source.title}</a>)}
      </div>}
    </div>
  </details>
}

function CorrectionCard({ correction }) {
  if (!correction?.answer) return null
  return <section className="correction-card">
    <p className="correction-label">Evidence-grounded correction</p>
    <p className="correction-answer">{correction.answer}</p>
    {correction.citations?.length > 0 && <div className="citations correction-citations"><span>Citations</span>
      {correction.citations.map((source, index) => source.url.startsWith('document://')
        ? <em key={`${source.url}-${index}`}>[{index + 1}] {source.title}</em>
        : <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">[{index + 1}] {source.title}</a>)}
    </div>}
  </section>
}

function MessageBubble({ message }) {
  if (message.role === 'user') return <article className="message user-message"><p>{message.content}</p></article>
  if (message.pending) return <article className="message assistant-message loading-message"><span className="assistant-avatar">V</span><div><p>Generating and checking sources<span className="typing-dots">...</span></p></div></article>
  return <article className="message assistant-message"><span className="assistant-avatar">V</span><div className="assistant-copy"><p>{message.content}</p>{message.model && <small>{message.model}</small>}<CorrectionCard correction={message.verification?.correction} /><VerificationCard result={message.verification} /></div></article>
}

function App() {
  const [conversations, setConversations] = useState(() => [createConversation()])
  const [activeId, setActiveId] = useState(() => conversations?.[0]?.id)
  const [draft, setDraft] = useState('')
  const [provider, setProvider] = useState('gemini')
  const [providers, setProviders] = useState([])
  const [apiStatus, setApiStatus] = useState('checking')
  const [verifyEnabled, setVerifyEnabled] = useState(true)
  const [evidenceMode, setEvidenceMode] = useState('web')
  const [document, setDocument] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [listening, setListening] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef(null)
  const recognitionRef = useRef(null)

  useEffect(() => {
    async function loadSetup() {
      try {
        const [healthResponse, providersResponse] = await Promise.all([fetch(`${API_URL}/health`), fetch(`${API_URL}/api/providers`)])
        if (!healthResponse.ok || !providersResponse.ok) throw new Error('Unavailable')
        const providerData = await providersResponse.json()
        setProviders(providerData)
        const firstConfigured = providerData.find((item) => item.id === 'gemini' && item.configured) ?? providerData.find((item) => item.configured)
        if (firstConfigured) setProvider(firstConfigured.id)
        setApiStatus('online')
      } catch { setApiStatus('offline') }
    }
    loadSetup()
    return () => recognitionRef.current?.stop()
  }, [])

  const activeConversation = useMemo(() => conversations.find((conversation) => conversation.id === activeId) ?? conversations[0], [activeId, conversations])

  function startNewChat() {
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setActiveId(conversation.id)
    setDraft('')
    setError('')
  }

  async function uploadPdf(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_URL}/api/documents`, { method: 'POST', body: formData })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'PDF upload failed.')
      setDocument(payload)
      setEvidenceMode('document')
    } catch (uploadError) { setError(uploadError.message) } finally { setUploading(false) }
  }

  function startListening() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition) { setError('Speech-to-text is not supported in this browser. Use Chrome or Edge.'); return }
    if (listening) { recognitionRef.current?.stop(); return }
    const recognition = new Recognition()
    recognition.lang = 'en-US'
    recognition.interimResults = true
    recognition.continuous = false
    recognition.onstart = () => { setListening(true); setError('') }
    recognition.onresult = (event) => {
      let transcript = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) transcript += event.results[index][0].transcript
      setDraft((current) => `${current}${current && !current.endsWith(' ') ? ' ' : ''}${transcript}`)
    }
    recognition.onerror = (event) => { setError(event.error === 'not-allowed' ? 'Microphone access was blocked. Allow it in your browser and try again.' : 'Voice transcription could not start.') }
    recognition.onend = () => setListening(false)
    recognitionRef.current = recognition
    recognition.start()
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const question = draft.trim()
    if (question.length < 3 || loading || !activeConversation) return
    if ((evidenceMode === 'document' || evidenceMode === 'hybrid') && !document) { setError('Attach a PDF before using PDF or hybrid evidence.'); return }

    const conversationId = activeConversation.id
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: question }
    const pendingMessage = { id: crypto.randomUUID(), role: 'assistant', pending: true }
    const history = activeConversation.messages.map(({ role, content }) => ({ role, content }))
    setConversations((current) => current.map((conversation) => conversation.id === conversationId ? { ...conversation, title: conversation.messages.length === 0 ? question.slice(0, 42) : conversation.title, messages: [...conversation.messages, userMessage, pendingMessage] } : conversation))
    setDraft('')
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, provider, mode: evidenceMode, verify: verifyEnabled, history, document_id: document?.id ?? null }) })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'The analysis request failed.')
      const assistantMessage = { id: crypto.randomUUID(), role: 'assistant', content: payload.answer ?? 'I could not generate an answer.', model: payload.model, verification: verifyEnabled ? payload : null }
      setConversations((current) => current.map((conversation) => conversation.id === conversationId ? { ...conversation, messages: conversation.messages.map((message) => message.id === pendingMessage.id ? assistantMessage : message) } : conversation))
    } catch (requestError) {
      setConversations((current) => current.map((conversation) => conversation.id === conversationId ? { ...conversation, messages: conversation.messages.filter((message) => message.id !== pendingMessage.id) } : conversation))
      setError(requestError.message)
    } finally { setLoading(false) }
  }

  return <main className="chat-app">
    <aside className="sidebar">
      <a className="brand" href="#top"><span className="brand-mark">V</span><span>VeriSight</span></a>
      <button className="new-chat" type="button" onClick={startNewChat}>+ New chat</button>
      <div className="history-heading"><span>History</span><small>This session</small></div>
      <nav className="chat-history" aria-label="Chat history">{conversations.map((conversation) => <button type="button" key={conversation.id} className={conversation.id === activeConversation?.id ? 'history-item active' : 'history-item'} onClick={() => setActiveId(conversation.id)}><span>{conversation.title}</span><small>{conversation.messages.length ? `${Math.ceil(conversation.messages.length / 2)} message${conversation.messages.length > 2 ? 's' : ''}` : 'Empty'}</small></button>)}</nav>
      <div className="sidebar-footer"><span className={`api-status ${apiStatus}`}><i></i> API {apiStatus}</span><p>Use web, PDF, or both as verification evidence.</p></div>
    </aside>
    <section className="chat-panel" id="top">
      <header className="chat-header">
        <div><p>AI HALLUCINATION DETECTION</p><h1>{activeConversation?.title ?? 'New conversation'}</h1></div>
        <div className="chat-controls">
          <label className="provider-select"><span>Model</span><select value={provider} onChange={(event) => setProvider(event.target.value)}>{providers.map((item) => <option key={item.id} value={item.id} disabled={!item.configured}>{item.label}{item.configured ? '' : ' (add key)'}</option>)}</select></label>
          <label className="provider-select"><span>Evidence</span><select value={evidenceMode} onChange={(event) => setEvidenceMode(event.target.value)}><option value="web">Web</option><option value="document" disabled={!document}>PDF</option><option value="hybrid" disabled={!document}>Hybrid</option></select></label>
          <label className="verify-toggle"><input type="checkbox" checked={verifyEnabled} onChange={(event) => setVerifyEnabled(event.target.checked)} /><span></span>Verify</label>
        </div>
      </header>
      <section className="message-thread" aria-live="polite">
        {!activeConversation?.messages.length && <div className="welcome-card"><span className="assistant-avatar large">V</span><div><h2>What would you like to know?</h2><p>Ask by typing or voice. Attach a PDF to verify answers against its contents.</p><div className="suggestions"><button type="button" onClick={() => setDraft('Who created the Python programming language?')}>Who created Python?</button><button type="button" onClick={() => setDraft('Explain quantum computing in simple terms.')}>Explain quantum computing</button></div></div></div>}
        {activeConversation?.messages.map((message) => <MessageBubble key={message.id} message={message} />)}
      </section>
      <form className="composer" onSubmit={handleSubmit}>
        {document && <div className="document-chip"><span>PDF: {document.filename} ({document.pages} page{document.pages === 1 ? '' : 's'})</span><button type="button" onClick={() => { setDocument(null); setEvidenceMode('web') }} aria-label="Remove PDF">×</button></div>}
        <div className="composer-row">
          <input ref={fileInputRef} type="file" accept="application/pdf" hidden onChange={uploadPdf} />
          <button className="utility-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={uploading || loading}>{uploading ? 'Uploading...' : 'Attach PDF'}</button>
          <button className={`utility-button ${listening ? 'listening' : ''}`} type="button" onClick={startListening} disabled={loading}>{listening ? 'Listening...' : 'Voice'}</button>
          <textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Message VeriSight..." aria-label="Message VeriSight" rows="1" />
          <button className="send-button" type="submit" disabled={loading || apiStatus !== 'online' || draft.trim().length < 3}>{loading ? 'Working...' : 'Send'}</button>
        </div>
        <p>{verifyEnabled ? `Verification is on: using ${evidenceMode === 'document' ? 'your PDF' : evidenceMode === 'hybrid' ? 'web and your PDF' : 'web evidence'}.` : 'Verification is off: this response will not receive a reliability score.'}</p>
        {error && <strong className="error">{error}</strong>}
      </form>
    </section>
  </main>
}

export default App
