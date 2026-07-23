import { useEffect, useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const statusLabel = {
  supported: 'Supported',
  uncertain: 'Needs review',
  unsupported: 'Unsupported',
}

function Score({ label, value, tone = 'indigo' }) {
  const percent = Math.round((value ?? 0) * 100)
  return <div className={`score-card ${tone}`}>
    <span>{label}</span>
    <strong>{percent}%</strong>
  </div>
}

function ClaimList({ claims }) {
  if (!claims?.length) return null
  return <section className="result-section">
    <div className="result-heading"><h3>Claim verification</h3><span>{claims.length} claims</span></div>
    <div className="claim-list">
      {claims.map((claim, index) => <article className="claim" key={`${claim.claim}-${index}`}>
        <span className={`claim-dot ${claim.status}`}></span>
        <div><b>{claim.claim}</b><p>{statusLabel[claim.status]} - {Math.round(claim.confidence * 100)}% verification confidence</p><small>{claim.rationale}</small></div>
      </article>)}
    </div>
  </section>
}

function EvidenceList({ evidence }) {
  if (!evidence?.length) return null
  return <section className="result-section evidence-section">
    <div className="result-heading"><h3>Evidence and citations</h3><span>{evidence.length} sources</span></div>
    {evidence.map((source, index) => <article className="evidence" key={`${source.url}-${index}`}>
      <span>[{index + 1}]</span><div><a href={source.url} target="_blank" rel="noreferrer">{source.title}</a></div>
    </article>)}
  </section>
}

function App() {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('web')
  const [provider, setProvider] = useState('evidence')
  const [providers, setProviders] = useState([])
  const [apiStatus, setApiStatus] = useState('checking')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    async function loadSetup() {
      try {
        const [healthResponse, providersResponse] = await Promise.all([
          fetch(`${API_URL}/health`),
          fetch(`${API_URL}/api/providers`),
        ])
        if (!healthResponse.ok || !providersResponse.ok) throw new Error('Unavailable')
        const providerData = await providersResponse.json()
        setProviders(providerData)
        setApiStatus('online')
      } catch {
        setApiStatus('offline')
      }
    }
    loadSetup()
  }, [])

  const configuredProviders = useMemo(
    () => providers.filter((item) => item.id !== 'evidence' && item.configured),
    [providers],
  )

  async function handleSubmit(event) {
    event.preventDefault()
    if (question.trim().length < 3) {
      setError('Enter a question with at least three characters.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim(), mode, provider }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'The analysis request failed.')
      setResult(payload)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  return <main className="app-shell">
    <nav className="topbar">
      <a className="brand" href="#top"><span className="brand-mark">V</span>VeriSight</a>
      <span className={`api-status ${apiStatus}`}><i></i> API {apiStatus}</span>
    </nav>

    <section className="hero" id="top">
      <p className="eyebrow">EVIDENCE-GROUNDED AI VERIFICATION</p>
      <h1>Trust AI answers<br /><em>with evidence.</em></h1>
      <p className="subtitle">Compare supported language models, verify claims against retrieved sources, and inspect explainable reliability signals.</p>
    </section>

    <section className="workspace">
      <form className="question-card" onSubmit={handleSubmit}>
        <div className="section-heading"><span>01</span><h2>Ask and verify</h2></div>
        <label htmlFor="question">Your question</label>
        <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="For example: Who created Python and when was it first released?" />
        <div className="select-grid">
          <div><label htmlFor="provider">LLM provider</label><select id="provider" value={provider} onChange={(event) => setProvider(event.target.value)}>
            {providers.map((item) => <option key={item.id} value={item.id} disabled={!item.configured}>{item.label}{item.configured ? '' : ' (add API key)'}</option>)}
            <option value="compare" disabled={configuredProviders.length < 2}>Compare configured LLMs{configuredProviders.length < 2 ? ' (requires 2 keys)' : ''}</option>
          </select></div>
          <div><label htmlFor="mode">Evidence mode</label><select id="mode" value={mode} onChange={(event) => setMode(event.target.value)}><option value="web">Web evidence</option><option value="document" disabled>Documents (next)</option><option value="hybrid" disabled>Hybrid (next)</option></select></div>
        </div>
        <button className="submit" type="submit" disabled={loading || apiStatus !== 'online'}>{loading ? 'Analyzing...' : 'Generate and verify ->'}</button>
        {providers.length > 0 && configuredProviders.length === 0 && <p className="setup-note">Add a provider key in <code>backend/.env</code> to generate LLM answers. The evidence-backed baseline works without a key.</p>}
        {error && <p className="error">{error}</p>}
      </form>

      <aside className="pipeline-card">
        <div className="section-heading"><span>02</span><h2>Verification pipeline</h2></div>
        <ol><li><b>Generate</b><small>One LLM or multiple selected providers answer.</small></li><li><b>Retrieve</b><small>Web evidence is collected once for the question.</small></li><li><b>Verify</b><small>Each claim is assessed against retrieved evidence.</small></li><li><b>Explain</b><small>Scores, citations, and comparison data are shown.</small></li></ol>
        <p className="pipeline-note">Next modules: PDF input, deeper retrieval, and uncertainty scoring.</p>
      </aside>
    </section>

    {result && <section className="results">
      <div className="results-top"><div><p className="eyebrow">ANALYSIS COMPLETE</p><h2>{result.provider === 'compare' ? 'LLM comparison' : 'Verification report'}</h2><p>{result.message}</p></div><Score label="Estimated reliability" value={result.reliability_score} /></div>
      {result.provider !== 'compare' && <section className="answer"><span>Generated answer</span><p>{result.answer}</p>{result.model && <small>Model: {result.model}</small>}</section>}
      {result.comparisons?.length > 0 && <section className="comparison-grid">{result.comparisons.map((item) => <article className="comparison" key={item.provider}><div><b>{item.provider}</b><span>{item.model}</span></div><Score label="Reliability" value={item.reliability_score} /><p>{item.answer}</p></article>)}</section>}
      <ClaimList claims={result.claims} />
      <EvidenceList evidence={result.evidence} />
    </section>}
  </main>
}

export default App
