import { useEffect, useState } from 'react'

const labels = ['supported', 'unsupported', 'uncertain']

const labelText = {
  supported: 'Supported',
  unsupported: 'Unsupported',
  uncertain: 'Needs review',
}

const percent = (value) => `${Math.round((value ?? 0) * 100)}%`

export default function EvaluationDashboard({ report, loading, error, onBack, onRefresh }) {
  const [selectedRunId, setSelectedRunId] = useState(null)
  const reports = report?.recent_reports?.length ? report.recent_reports : report ? [report] : []

  useEffect(() => {
    setSelectedRunId(report?.run_id ?? null)
  }, [report?.run_id])

  const selectedReport = reports.find((item) => item.run_id === selectedRunId) ?? report
  const metrics = selectedReport?.metrics

  return <section className="evaluation-dashboard" aria-labelledby="evaluation-title">
    <header className="evaluation-header">
      <div><p>LOCAL RESEARCH VIEW</p><h2 id="evaluation-title">Verification evaluation</h2><span>Benchmark metrics for the project team. These results are not shown in normal user chats.</span></div>
      <div className="evaluation-actions"><button type="button" onClick={onRefresh} disabled={loading}>Refresh</button><button type="button" onClick={onBack}>Back to chat</button></div>
    </header>
    {loading && <p className="evaluation-state">Loading the latest evaluation report...</p>}
    {error && <p className="evaluation-state error">{error}</p>}
    {metrics && <>
      <section className="evaluation-section report-history">
        <div><h3>Recent benchmark reports</h3><p>New reports appear after running the evaluation script. Backend unit tests do not produce benchmark metrics.</p></div>
        <div className="report-list">{reports.map((item) => <button type="button" key={item.run_id} className={item.run_id === selectedReport?.run_id ? 'report-item active' : 'report-item'} onClick={() => setSelectedRunId(item.run_id)}><strong>{item.metrics.dataset ?? 'Local benchmark'}</strong><span>{item.run_id} · {item.metrics.cases ?? 0} claims · {percent(item.metrics.accuracy)} accuracy</span></button>)}</div>
      </section>
      <div className="metric-grid">
        <article><span>Accuracy</span><strong>{percent(metrics.accuracy)}</strong></article>
        <article><span>Macro F1</span><strong>{percent(metrics.macro_f1)}</strong></article>
        <article><span>Claims evaluated</span><strong>{metrics.cases ?? 0}</strong></article>
        <article><span>Avg. latency</span><strong>{Math.round(metrics.average_latency_ms ?? 0)} ms</strong></article>
      </div>
      <section className="evaluation-section"><h3>Verdict metrics</h3><div className="evaluation-table-wrap"><table><thead><tr><th>Verdict</th><th>Cases</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{labels.map((label) => { const item = metrics.per_label?.[label] ?? {}; return <tr key={label}><td>{labelText[label]}</td><td>{item.support ?? 0}</td><td>{percent(item.precision)}</td><td>{percent(item.recall)}</td><td>{percent(item.f1)}</td></tr> })}</tbody></table></div></section>
      <section className="evaluation-section"><h3>Confusion matrix</h3><p>Rows are expected labels; columns are the verifier’s predicted labels.</p><div className="matrix"><div></div>{labels.map((label) => <b key={`heading-${label}`}>{labelText[label]}</b>)}{labels.flatMap((truth) => [<b key={`row-${truth}`}>{labelText[truth]}</b>, ...labels.map((guess) => <span key={`${truth}-${guess}`}>{metrics.confusion_matrix?.[truth]?.[guess] ?? 0}</span>)])}</div></section>
      <footer className="evaluation-footer">Run: {selectedReport?.run_id} · Dataset: {metrics.dataset ?? 'local benchmark'} · Generated: {metrics.generated_at ? new Date(metrics.generated_at).toLocaleString() : '—'}</footer>
    </>}
  </section>
}
