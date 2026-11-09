import React, { useState, useEffect, useRef } from 'react'
import {
  Sparkles,
  Globe,
  FileText,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  Download,
  Zap,
  Cpu,
  ArrowRight,
  RefreshCw,
  FolderArchive,
  X,
  FileCode,
  Tag,
} from 'lucide-react'
import { marked } from 'marked'

interface HealthState {
  status: string
  version: string
  api_key_configured: boolean
  default_model: string
  supported_models_count: number
}

interface ModelItem {
  id: string
  name: string
  description: string
  preference: string
  intended_use: string
  max_output_tokens: number
  is_default: boolean
}

interface StructuredOutput {
  title: string
  tldr: string
  summary: string
  key_takeaways: string[]
  topics_and_keywords: string[]
  central_message: string
}

interface Metrics {
  total_duration_sec?: number
  first_chunk_latency_sec?: number
  first_chunk_latency_ms?: number
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  tokens_per_second?: number
  input_word_count?: number
  generated_word_count?: number
}

interface HistoryItem {
  filename: string
  path: string
  size_bytes: number
  modified_at: string
  title?: string
}

const SAMPLE_ARTICLE = `Artificial intelligence has entered a transformative era driven by large language models (LLMs) and multimodal foundation models. Unlike traditional task-specific machine learning models, foundation models are pre-trained on vast datasets and fine-tuned for a multitude of downstream tasks, ranging from natural language processing to computer vision and robotics.

Recent breakthroughs in transformer architectures, attention mechanisms, and scaling laws have substantially expanded model capabilities. However, with this unprecedented scale comes a critical engineering challenge: processing speed, computational cost, and context length limitations. In production environments, enterprise applications often face latency constraints when ingesting long-form technical reports, financial filings, and academic literature.

To overcome these challenges, advanced AI architectures utilize Map-Reduce workflows and hierarchical chunking strategies. In a Map-Reduce summarization pipeline, extensive documents are split into coherent, overlapping chunks. Each chunk is processed in parallel by lightweight flash models (the "Map" step), extracting salient points. Subsequently, an aggregator model synthesizes these interim outputs into a structured executive brief (the "Reduce" step). This methodology guarantees that critical context is never truncated, hallucinations are curtailed, and end-to-end processing latency remains strictly manageable.`

export default function App() {
  // Navigation / Tabs
  const [inputTab, setInputTab] = useState<'url' | 'text' | 'file'>('url')
  
  // Form State
  const [urlInput, setUrlInput] = useState('')
  const [textInput, setTextInput] = useState('')
  const [titleHint, setTitleHint] = useState('')
  const [selectedModel, setSelectedModel] = useState('gemini-3.5-flash')
  const [outputFormat, setOutputFormat] = useState<'markdown' | 'json'>('markdown')
  const [enableStreaming, setEnableStreaming] = useState(true)
  
  // API & Metadata State
  const [health, setHealth] = useState<HealthState | null>(null)
  const [models, setModels] = useState<ModelItem[]>([])
  
  // UI Status
  const [isExtracting, setIsExtracting] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [errorHint, setErrorHint] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  
  // Output State
  const [summaryText, setSummaryText] = useState('')
  const [structuredData, setStructuredData] = useState<StructuredOutput | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [savedFilePath, setSavedFilePath] = useState<string | null>(null)
  const [activeModelUsed, setActiveModelUsed] = useState<string>('')
  
  // History Drawer
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
  const [loadingHistory, setLoadingHistory] = useState(false)

  const summaryEndRef = useRef<HTMLDivElement>(null)

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) {
        const data: HealthState = await res.json()
        setHealth(data)
        if (data.default_model) {
          setSelectedModel(data.default_model)
        }
      }
    } catch (err) {
      console.warn('API health check error:', err)
    }
  }

  const fetchModels = async () => {
    try {
      const res = await fetch('/api/models')
      if (res.ok) {
        const data: ModelItem[] = await res.json()
        setModels(data)
      }
    } catch (err) {
      console.warn('API models fetch error:', err)
    }
  }

  // Fetch health and models on load
  useEffect(() => {
    fetchHealth()
    fetchModels()
  }, [])

  const fetchHistory = async () => {
    setLoadingHistory(true)
    try {
      const res = await fetch('/api/history')
      if (res.ok) {
        const data: HistoryItem[] = await res.json()
        setHistoryItems(data)
      }
    } catch (e) {
      console.warn('Failed to load history:', e)
    } finally {
      setLoadingHistory(false)
    }
  }

  // URL Extraction Handler
  const handleExtractUrl = async () => {
    if (!urlInput.trim()) {
      setErrorMsg('Please enter a valid website URL.')
      setErrorHint('e.g. https://example.com/article')
      return
    }

    setIsExtracting(true)
    setErrorMsg(null)
    setErrorHint(null)

    try {
      const res = await fetch('/api/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInput.trim() }),
      })

      const data = await res.json()
      if (!res.ok) {
        setErrorMsg(data.error || 'Failed to extract article.')
        setErrorHint(data.hint || 'Check if the URL is accessible and allows web extraction.')
        return
      }

      setTextInput(data.text)
      if (data.title) {
        setTitleHint(data.title)
      }
      setInputTab('text')
    } catch (err: any) {
      setErrorMsg('Could not reach backend extraction service.')
      setErrorHint(err?.message || 'Ensure the FastAPI backend server is running.')
    } finally {
      setIsExtracting(false)
    }
  }

  // File Upload Handler
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setErrorMsg(null)
    setErrorHint(null)
    setIsExtracting(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()
      if (!res.ok) {
        setErrorMsg(data.error || data.detail || 'Failed to upload file.')
        setErrorHint('Only .txt and .md files are supported.')
        return
      }

      setTextInput(data.text)
      setTitleHint(data.title)
      setInputTab('text')
    } catch (err: any) {
      setErrorMsg('File upload failed.')
      setErrorHint(err?.message || 'Ensure your file is valid UTF-8 text.')
    } finally {
      setIsExtracting(false)
      // reset file input
      e.target.value = ''
    }
  }

  // Summarize Handler (Batch or Stream)
  const handleSummarize = async () => {
    if (!textInput.trim()) {
      setErrorMsg('Please provide article content to summarize.')
      setErrorHint('Paste text, enter a URL, or upload a file first.')
      return
    }

    const wordCount = textInput.trim().split(/\s+/).length
    if (wordCount < 30) {
      setErrorMsg(`Article is too short (${wordCount} words). Minimum required is 30 words.`)
      setErrorHint('Please provide a longer text or article for meaningful summarization.')
      return
    }

    setIsSummarizing(true)
    setErrorMsg(null)
    setErrorHint(null)
    setSummaryText('')
    setStructuredData(null)
    setMetrics(null)
    setSavedFilePath(null)
    setActiveModelUsed(selectedModel)

    if (enableStreaming && outputFormat === 'markdown') {
      // Use SSE Streaming Endpoint
      try {
        const response = await fetch('/api/summarize/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: textInput,
            title_hint: titleHint || undefined,
            url: urlInput || undefined,
            model_id: selectedModel,
            output_format: 'markdown',
            save_to_disk: true,
          }),
        })

        if (!response.ok) {
          const errData = await response.json()
          setErrorMsg(errData.error || 'Summarization failed.')
          setErrorHint(errData.hint)
          setIsSummarizing(false)
          return
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let accumulated = ''

        if (reader) {
          let buffer = ''
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const eventData = JSON.parse(line.substring(6))
                  if (eventData.event === 'chunk') {
                    accumulated += eventData.text
                    setSummaryText(accumulated)
                  } else if (eventData.event === 'done') {
                    setSummaryText(eventData.summary)
                    setMetrics(eventData.metrics)
                    setSavedFilePath(eventData.saved_file)
                  } else if (eventData.event === 'error') {
                    setErrorMsg(eventData.error)
                  }
                } catch (parseErr) {
                  console.error('SSE JSON parse error:', parseErr)
                }
              }
            }
          }
        }
      } catch (err: any) {
        setErrorMsg('Network error while streaming summary.')
        setErrorHint(err.message)
      } finally {
        setIsSummarizing(false)
      }
    } else {
      // Standard Batch Endpoint (also handles structured JSON format)
      try {
        const res = await fetch('/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: textInput,
            title_hint: titleHint || undefined,
            url: urlInput || undefined,
            model_id: selectedModel,
            output_format: outputFormat,
            save_to_disk: true,
          }),
        })

        const data = await res.json()
        if (!res.ok) {
          setErrorMsg(data.error || 'Summarization failed.')
          setErrorHint(data.hint)
          return
        }

        setSummaryText(data.summary)
        setStructuredData(data.structured || null)
        setMetrics(data.metrics || null)
        setSavedFilePath(data.saved_file || null)
        setActiveModelUsed(data.model_used)
      } catch (err: any) {
        setErrorMsg('Failed to complete summarization.')
        setErrorHint(err.message)
      } finally {
        setIsSummarizing(false)
      }
    }
  }

  // Copy Summary to Clipboard
  const handleCopy = () => {
    const textToCopy = structuredData
      ? `# ${structuredData.title}\n\n**TL;DR:** ${structuredData.tldr}\n\n## Core Thesis\n${structuredData.central_message}\n\n## Key Takeaways\n${structuredData.key_takeaways.map((t) => `- ${t}`).join('\n')}\n\n## Summary\n${structuredData.summary}`
      : summaryText

    navigator.clipboard.writeText(textToCopy)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Download Summary as Markdown or JSON
  const handleDownload = (format: 'md' | 'json') => {
    let content = summaryText
    let filename = `summary_${new Date().toISOString().slice(0, 10)}.${format}`
    let mime = 'text/markdown'

    if (format === 'json') {
      content = JSON.stringify(structuredData || { summary: summaryText, metrics }, null, 2)
      mime = 'application/json'
    }

    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // Estimated stats for input
  const inputWords = textInput.trim() ? textInput.trim().split(/\s+/).length : 0
  const inputChars = textInput.length
  const estimatedTokens = Math.max(0, Math.round(inputWords * 1.33))
  const isMapReduceNeeded = inputWords > 3500

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '2rem 1.5rem 5rem' }}>
      {/* Top Header */}
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem',
          paddingBottom: '2rem',
          borderBottom: '1px solid var(--border-color)',
          marginBottom: '2rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 8px 20px rgba(99, 102, 241, 0.35)',
            }}
          >
            <Sparkles size={24} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              AI Article Summarizer
              <span
                style={{
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  padding: '2px 8px',
                  borderRadius: '12px',
                  background: 'var(--primary-light)',
                  color: '#818cf8',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                }}
              >
                FastAPI + Vite
              </span>
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
              Precision Document Synthesis powered by Google Gemini
            </p>
          </div>
        </div>

        {/* Right Status Badges & History Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '20px',
              background: health?.api_key_configured ? 'var(--success-bg)' : 'var(--danger-bg)',
              border: `1px solid ${health?.api_key_configured ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
              fontSize: '0.8rem',
              fontWeight: 500,
            }}
          >
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: health?.api_key_configured ? '#10b981' : '#ef4444',
              }}
            />
            {health?.api_key_configured ? 'Gemini API Ready' : 'API Key Missing in .env'}
          </div>

          <button
            onClick={() => {
              setIsHistoryOpen(true)
              fetchHistory()
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '7px 14px',
              borderRadius: '8px',
              background: 'var(--bg-card-subtle)',
              border: '1px solid var(--border-color)',
              color: 'var(--text-primary)',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 500,
              transition: 'all 0.2s',
            }}
          >
            <FolderArchive size={16} color="#818cf8" />
            Saved Outputs
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '2rem' }}>
        {/* Left Column: Input Source & Config */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Input Tabs Card */}
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.25rem',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
            }}
          >
            {/* Tabs Header */}
            <div
              style={{
                display: 'flex',
                background: 'var(--bg-input)',
                borderRadius: 'var(--radius-md)',
                padding: '4px',
                marginBottom: '1.25rem',
                border: '1px solid var(--border-color)',
              }}
            >
              {[
                { id: 'url', label: 'Article URL', icon: Globe },
                { id: 'text', label: 'Direct Text', icon: FileText },
                { id: 'file', label: 'File Upload', icon: UploadCloud },
              ].map((tab) => {
                const Icon = tab.icon
                const isActive = inputTab === tab.id
                return (
                  <button
                    key={tab.id}
                    onClick={() => setInputTab(tab.id as any)}
                    style={{
                      flex: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      border: 'none',
                      background: isActive ? 'var(--primary)' : 'transparent',
                      color: isActive ? '#ffffff' : 'var(--text-secondary)',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <Icon size={16} />
                    {tab.label}
                  </button>
                )
              })}
            </div>

            {/* URL Input Form */}
            {inputTab === 'url' && (
              <div>
                <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                  Webpage Article URL
                </label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    type="url"
                    placeholder="https://news.ycombinator.com/item?id=... or blog article"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleExtractUrl()}
                    style={{
                      flex: 1,
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)',
                      padding: '0.75rem 1rem',
                      color: 'var(--text-primary)',
                      fontSize: '0.9rem',
                      outline: 'none',
                    }}
                  />
                  <button
                    onClick={handleExtractUrl}
                    disabled={isExtracting}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '0.75rem 1.25rem',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--primary)',
                      border: 'none',
                      color: '#ffffff',
                      fontWeight: 600,
                      cursor: isExtracting ? 'not-allowed' : 'pointer',
                      opacity: isExtracting ? 0.7 : 1,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {isExtracting ? (
                      <>
                        <RefreshCw size={16} className="animate-spin" />
                        Fetching...
                      </>
                    ) : (
                      <>
                        Extract
                        <ArrowRight size={16} />
                      </>
                    )}
                  </button>
                </div>
                <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Respects robots.txt policies, strips navigation and ads, and extracts core article prose.
                </p>
              </div>
            )}

            {/* Direct Text Input */}
            {inputTab === 'text' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    Article Content
                  </label>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      onClick={() => {
                        setTextInput(SAMPLE_ARTICLE)
                        setTitleHint('Transformative Scaling & Map-Reduce in LLMs')
                      }}
                      style={{
                        background: 'transparent',
                        border: '1px solid var(--border-color)',
                        padding: '3px 8px',
                        borderRadius: '4px',
                        color: '#818cf8',
                        fontSize: '0.75rem',
                        cursor: 'pointer',
                      }}
                    >
                      Load Sample
                    </button>
                    {textInput && (
                      <button
                        onClick={() => {
                          setTextInput('')
                          setTitleHint('')
                        }}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: 'var(--text-muted)',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                        }}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>

                <textarea
                  rows={9}
                  placeholder="Paste article body text here..."
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                    padding: '0.75rem 1rem',
                    color: 'var(--text-primary)',
                    fontSize: '0.9rem',
                    lineHeight: 1.5,
                    resize: 'vertical',
                    outline: 'none',
                  }}
                />

                <div style={{ marginTop: '0.75rem' }}>
                  <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                    Optional Title Hint
                  </label>
                  <input
                    type="text"
                    placeholder="Article title or topic..."
                    value={titleHint}
                    onChange={(e) => setTitleHint(e.target.value)}
                    style={{
                      width: '100%',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)',
                      padding: '0.5rem 0.75rem',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem',
                      outline: 'none',
                    }}
                  />
                </div>
              </div>
            )}

            {/* File Upload Tab */}
            {inputTab === 'file' && (
              <div>
                <label
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '2.5rem 1.5rem',
                    border: '2px dashed var(--border-color)',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-input)',
                    cursor: 'pointer',
                    transition: 'border-color 0.2s',
                  }}
                >
                  <UploadCloud size={36} color="#818cf8" style={{ marginBottom: '0.75rem' }} />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Click to select or drop article file</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Supports plain text (.txt) and Markdown (.md) documents
                  </span>
                  <input
                    type="file"
                    accept=".txt,.md,.markdown"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>
            )}

            {/* Source Content Counter Bar */}
            {inputWords > 0 && (
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '1rem',
                  paddingTop: '0.75rem',
                  borderTop: '1px solid var(--border-color)',
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  flexWrap: 'wrap',
                  gap: '0.5rem',
                }}
              >
                <div style={{ display: 'flex', gap: '1rem' }}>
                  <span>
                    <strong>{inputWords.toLocaleString()}</strong> words
                  </span>
                  <span>
                    <strong>{inputChars.toLocaleString()}</strong> chars
                  </span>
                  <span>
                    ~<strong>{estimatedTokens.toLocaleString()}</strong> tokens
                  </span>
                </div>

                {isMapReduceNeeded && (
                  <span
                    style={{
                      background: 'var(--warning-bg)',
                      color: '#fbbf24',
                      padding: '2px 8px',
                      borderRadius: '6px',
                      border: '1px solid rgba(245, 158, 11, 0.3)',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                    }}
                  >
                    ⚡ Map-Reduce Required (&gt;3,500 words)
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Model & Output Configuration */}
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.25rem',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
            }}
          >
            <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={18} color="#818cf8" />
              Engine & Synthesis Settings
            </h2>

            {/* Model Select */}
            <div style={{ marginBottom: '1.25rem' }}>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Gemini Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{
                  width: '100%',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.65rem 0.85rem',
                  color: 'var(--text-primary)',
                  fontSize: '0.9rem',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              >
                {models.length > 0 ? (
                  models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} — {m.preference} ({m.intended_use})
                    </option>
                  ))
                ) : (
                  <>
                    <option value="gemini-3.5-flash">Gemini 3.5 Flash (Balanced Default)</option>
                    <option value="gemini-3.5-flash-lite">Gemini 3.5 Flash-Lite (Fastest)</option>
                    <option value="gemini-3.8-flash">Gemini 3.8 Flash (High Reasoning Depth)</option>
                    <option value="gemini-3.6-flash">Gemini 3.6 Flash (Standard)</option>
                    <option value="gemini-3.1-flash-lite">Gemini 3.1 Flash-Lite (Lightweight)</option>
                  </>
                )}
              </select>
            </div>

            {/* Format & Mode Controls */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                  Output Presentation
                </label>
                <div style={{ display: 'flex', background: 'var(--bg-input)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <button
                    onClick={() => setOutputFormat('markdown')}
                    style={{
                      flex: 1,
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      background: outputFormat === 'markdown' ? 'var(--primary)' : 'transparent',
                      color: outputFormat === 'markdown' ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Markdown
                  </button>
                  <button
                    onClick={() => {
                      setOutputFormat('json')
                      setEnableStreaming(false)
                    }}
                    style={{
                      flex: 1,
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      background: outputFormat === 'json' ? 'var(--primary)' : 'transparent',
                      color: outputFormat === 'json' ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    JSON Cards
                  </button>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.4rem' }}>
                  Delivery Mode
                </label>
                <div style={{ display: 'flex', background: 'var(--bg-input)', padding: '3px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <button
                    onClick={() => setEnableStreaming(true)}
                    disabled={outputFormat === 'json'}
                    style={{
                      flex: 1,
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      background: enableStreaming && outputFormat === 'markdown' ? 'var(--primary)' : 'transparent',
                      color: enableStreaming && outputFormat === 'markdown' ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: outputFormat === 'json' ? 'not-allowed' : 'pointer',
                      opacity: outputFormat === 'json' ? 0.4 : 1,
                    }}
                  >
                    ⚡ Stream
                  </button>
                  <button
                    onClick={() => setEnableStreaming(false)}
                    style={{
                      flex: 1,
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      background: !enableStreaming ? 'var(--primary)' : 'transparent',
                      color: !enableStreaming ? '#ffffff' : 'var(--text-muted)',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    Batch
                  </button>
                </div>
              </div>
            </div>

            {/* Summarize Action Button */}
            <button
              onClick={handleSummarize}
              disabled={isSummarizing || !textInput.trim()}
              style={{
                width: '100%',
                marginTop: '1.25rem',
                padding: '0.85rem',
                borderRadius: 'var(--radius-md)',
                background: 'linear-gradient(135deg, #6366f1, #4f46e5)',
                border: 'none',
                color: '#ffffff',
                fontSize: '1rem',
                fontWeight: 700,
                cursor: isSummarizing || !textInput.trim() ? 'not-allowed' : 'pointer',
                opacity: isSummarizing || !textInput.trim() ? 0.6 : 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                boxShadow: '0 6px 20px rgba(99, 102, 241, 0.3)',
                transition: 'all 0.2s',
              }}
            >
              {isSummarizing ? (
                <>
                  <RefreshCw size={18} className="animate-spin" />
                  Synthesizing with {selectedModel}...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Generate AI Summary
                </>
              )}
            </button>
          </div>

          {/* Error Banner */}
          {errorMsg && (
            <div
              style={{
                background: 'var(--danger-bg)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                borderRadius: 'var(--radius-md)',
                padding: '1rem',
                color: '#fca5a5',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 700, fontSize: '0.9rem' }}>
                <AlertCircle size={18} color="#ef4444" />
                {errorMsg}
              </div>
              {errorHint && (
                <div style={{ marginTop: '0.35rem', fontSize: '0.8rem', color: '#f87171', paddingLeft: '26px' }}>
                  💡 {errorHint}
                </div>
              )}
            </div>
          )}
        </section>

        {/* Right Column: Generated Summary & Performance Metrics */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.5rem',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)',
              minHeight: '480px',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {/* Header & Actions */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                paddingBottom: '1rem',
                borderBottom: '1px solid var(--border-color)',
                marginBottom: '1.25rem',
              }}
            >
              <div>
                <h2 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <FileCode size={20} color="#818cf8" />
                  Synthesis Output
                </h2>
                {activeModelUsed && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Generated by {activeModelUsed}
                  </span>
                )}
              </div>

              {/* Action Buttons */}
              {(summaryText || structuredData) && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <button
                    onClick={handleCopy}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      background: 'var(--bg-card-subtle)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-primary)',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                    }}
                  >
                    {copied ? <Check size={14} color="#10b981" /> : <Copy size={14} />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>

                  <button
                    onClick={() => handleDownload('md')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                      background: 'var(--bg-card-subtle)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-primary)',
                      padding: '6px 10px',
                      borderRadius: '6px',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                    }}
                  >
                    <Download size={14} />
                    .md
                  </button>

                  {structuredData && (
                    <button
                      onClick={() => handleDownload('json')}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        background: 'var(--bg-card-subtle)',
                        border: '1px solid var(--border-color)',
                        color: 'var(--text-primary)',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        fontSize: '0.8rem',
                        cursor: 'pointer',
                      }}
                    >
                      <Download size={14} />
                      .json
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Empty State */}
            {!summaryText && !structuredData && !isSummarizing && (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-muted)',
                  textAlign: 'center',
                  padding: '3rem 1rem',
                }}
              >
                <div
                  style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '50%',
                    background: 'var(--bg-input)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1rem',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  <Zap size={28} color="#6366f1" />
                </div>
                <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                  No Summary Generated Yet
                </h3>
                <p style={{ fontSize: '0.85rem', maxWidth: '340px' }}>
                  Input an article via URL, paste direct text, or upload a markdown/text file, then click "Generate AI Summary".
                </p>
              </div>
            )}

            {/* Streaming Active Indicator */}
            {isSummarizing && !summaryText && (
              <div
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-secondary)',
                  gap: '1rem',
                }}
              >
                <RefreshCw size={36} color="#818cf8" className="animate-spin" />
                <p style={{ fontSize: '0.9rem', fontWeight: 500 }}>
                  Analyzing structure and generating summary...
                </p>
              </div>
            )}

            {/* Structured JSON Cards Mode */}
            {structuredData && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {/* Title & Core Thesis */}
                <div style={{ background: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.35rem' }}>
                    {structuredData.title}
                  </h3>
                  <p style={{ fontSize: '0.85rem', color: '#a5b4fc', fontStyle: 'italic' }}>
                    "{structuredData.central_message}"
                  </p>
                </div>

                {/* TL;DR Hero Box */}
                <div
                  style={{
                    background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(6, 182, 212, 0.12))',
                    border: '1px solid rgba(99, 102, 241, 0.35)',
                    borderRadius: 'var(--radius-md)',
                    padding: '1rem 1.25rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#818cf8', fontWeight: 700, fontSize: '0.85rem', marginBottom: '0.35rem' }}>
                    <Zap size={16} />
                    TL;DR
                  </div>
                  <p style={{ fontSize: '0.95rem', color: '#f1f5f9', lineHeight: 1.6 }}>
                    {structuredData.tldr}
                  </p>
                </div>

                {/* Key Takeaways */}
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.65rem' }}>
                    Key Takeaways
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                    {structuredData.key_takeaways.map((takeaway, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: 'flex',
                          alignItems: 'flex-start',
                          gap: '10px',
                          background: 'var(--bg-card-subtle)',
                          padding: '0.85rem 1rem',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-color)',
                        }}
                      >
                        <span
                          style={{
                            minWidth: '22px',
                            height: '22px',
                            borderRadius: '50%',
                            background: 'var(--primary)',
                            color: '#ffffff',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                          }}
                        >
                          {idx + 1}
                        </span>
                        <p style={{ fontSize: '0.9rem', color: '#e2e8f0', lineHeight: 1.5 }}>
                          {takeaway}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* In-depth Summary */}
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                    Executive Summary
                  </h4>
                  <p style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: 1.65, background: 'var(--bg-input)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                    {structuredData.summary}
                  </p>
                </div>

                {/* Topics / Tags */}
                {structuredData.topics_and_keywords.length > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <Tag size={14} color="#64748b" />
                    {structuredData.topics_and_keywords.map((kw, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: '0.75rem',
                          padding: '3px 9px',
                          borderRadius: '12px',
                          background: 'var(--bg-card-subtle)',
                          border: '1px solid var(--border-color)',
                          color: '#94a3b8',
                        }}
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Markdown Rendered Mode (or Live Stream Output) */}
            {!structuredData && summaryText && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div
                  className="markdown-body"
                  dangerouslySetInnerHTML={{ __html: marked.parse(summaryText) as string }}
                />
                <div ref={summaryEndRef} />
              </div>
            )}

            {/* Metrics Footer Dashboard */}
            {metrics && (
              <div
                style={{
                  marginTop: 'auto',
                  paddingTop: '1.25rem',
                  borderTop: '1px solid var(--border-color)',
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                  gap: '0.75rem',
                }}
              >
                <div style={{ background: 'var(--bg-input)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Latency</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#38bdf8' }}>
                    {metrics.total_duration_sec}s
                  </div>
                </div>

                {metrics.first_chunk_latency_ms && (
                  <div style={{ background: 'var(--bg-input)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TTFT</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#818cf8' }}>
                      {metrics.first_chunk_latency_ms}ms
                    </div>
                  </div>
                )}

                {metrics.prompt_tokens !== undefined && metrics.prompt_tokens !== null && (
                  <div style={{ background: 'var(--bg-input)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Tokens (In / Out)</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>
                      {metrics.prompt_tokens} / {metrics.completion_tokens || '-'}
                    </div>
                  </div>
                )}

                {metrics.input_word_count !== undefined && metrics.generated_word_count !== undefined && (
                  <div style={{ background: 'var(--bg-input)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Words Saved</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981' }}>
                      {Math.max(0, metrics.input_word_count - metrics.generated_word_count).toLocaleString()}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Saved File Notice */}
            {savedFilePath && (
              <div
                style={{
                  marginTop: '0.75rem',
                  fontSize: '0.75rem',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <CheckCircle2 size={13} color="#10b981" />
                Saved to outputs directory: <code style={{ color: '#94a3b8' }}>{savedFilePath}</code>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* History Drawer Modal */}
      {isHistoryOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.7)',
            backdropFilter: 'blur(4px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
            zIndex: 9999,
          }}
          onClick={() => setIsHistoryOpen(false)}
        >
          <div
            style={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-lg)',
              width: '100%',
              maxWidth: '640px',
              maxHeight: '80vh',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '1.25rem',
                borderBottom: '1px solid var(--border-color)',
              }}
            >
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FolderArchive size={20} color="#818cf8" />
                Saved Summaries Archive
              </h3>
              <button
                onClick={() => setIsHistoryOpen(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ padding: '1.25rem', overflowY: 'auto', flex: 1 }}>
              {loadingHistory ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  Loading saved files...
                </div>
              ) : historyItems.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                  No saved summaries found in outputs/ yet.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {historyItems.map((item) => (
                    <div
                      key={item.filename}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        background: 'var(--bg-input)',
                        padding: '0.85rem 1rem',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-color)',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                          {item.title || item.filename}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                          {item.modified_at} • {(item.size_bytes / 1024).toFixed(1)} KB
                        </div>
                      </div>

                      <a
                        href={`/api/history/${item.filename}`}
                        download={item.filename}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          background: 'var(--primary)',
                          color: '#ffffff',
                          padding: '6px 12px',
                          borderRadius: '6px',
                          fontSize: '0.78rem',
                          fontWeight: 600,
                          textDecoration: 'none',
                        }}
                      >
                        <Download size={14} />
                        Download
                      </a>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
