import { useCallback, useEffect, useRef, useState } from 'react'
import { api, CameraPreset, DetectionLog, Person, StreamStatus } from '../api/client'
import { EmptyState, Panel, PanelHeader, PrimaryButton, SecondaryButton, DangerButton } from '../components/ui'

export default function Livestream() {
  const [rtspUrl, setRtspUrl] = useState('webcam:0')
  const [location, setLocation] = useState('Sector 7G - Perimeter')
  const [connected, setConnected] = useState(false)
  const [status, setStatus] = useState<StreamStatus | null>(null)
  const [persons, setPersons] = useState<Person[]>([])
  const [activeTargets, setActiveTargets] = useState<Set<string>>(new Set())
  const [logs, setLogs] = useState<DetectionLog[]>([])
  const [error, setError] = useState('')
  const [presets, setPresets] = useState<CameraPreset[]>([])
  const [connecting, setConnecting] = useState(false)
  const [frameUrl, setFrameUrl] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const frameWsRef = useRef<WebSocket | null>(null)
  const frameUrlRef = useRef('')
  const lastSeqRef = useRef(-1)

  const loadPersons = useCallback(async () => {
    const data = await api.listPersons()
    setPersons(data.filter((p) => !p.is_unknown))
    setActiveTargets(new Set(data.filter((p) => !p.is_unknown).map((p) => p.name)))
  }, [])

  const loadLogs = useCallback(async () => {
    const data = await api.getLogs(50, 'livestream')
    setLogs(data)
  }, [])

  const stopFrameStream = useCallback(() => {
    frameWsRef.current?.close()
    frameWsRef.current = null
    if (frameUrlRef.current) {
      URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = ''
    }
    setFrameUrl('')
    lastSeqRef.current = -1
  }, [])

  const startFrameStream = useCallback(() => {
    if (frameWsRef.current && frameWsRef.current.readyState <= WebSocket.OPEN) return
    stopFrameStream()
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/api/stream/ws/frames`)
    ws.binaryType = 'arraybuffer'
    let pending: ArrayBuffer | null = null
    let raf = 0
    const flush = () => {
      raf = 0
      if (!pending) return
      const buf = pending
      pending = null
      const view = new DataView(buf)
      const seq = view.getUint32(0)
      if (seq === lastSeqRef.current) return
      lastSeqRef.current = seq
      const blob = new Blob([buf.slice(4)], { type: 'image/jpeg' })
      const url = URL.createObjectURL(blob)
      if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current)
      frameUrlRef.current = url
      setFrameUrl(url)
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') return
      const buf = ev.data as ArrayBuffer
      if (buf.byteLength < 5) return
      pending = buf
      if (!raf) raf = requestAnimationFrame(flush)
    }
    ws.onerror = () => {
      stopFrameStream()
    }
    frameWsRef.current = ws
  }, [stopFrameStream])

  const connectWs = useCallback(() => {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${window.location.host}/api/stream/ws`)
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data)
      if (data.type === 'ping') return
      setLogs((prev) => [data, ...prev].slice(0, 50))
    }
    wsRef.current = ws
  }, [])

  useEffect(() => {
    loadPersons()
    loadLogs()
    api.listStreamPresets().then(setPresets).catch(() => {})
    api.streamStatus().then((s) => {
      setConnected(s.connected)
      setStatus(s)
      if (s.connected) {
        connectWs()
        startFrameStream()
      }
    }).catch(() => {})
    return () => {
      wsRef.current?.close()
      stopFrameStream()
    }
  }, [loadPersons, loadLogs, connectWs, startFrameStream, stopFrameStream])

  useEffect(() => {
    if (!connected) return
    const id = setInterval(() => {
      api.streamStatus().then(setStatus).catch(() => {})
    }, 2000)
    return () => clearInterval(id)
  }, [connected])

  const handleConnect = async () => {
    setError('')
    setConnecting(true)
    try {
      const s = await api.connectStream(rtspUrl, location, Array.from(activeTargets))
      setConnected(true)
      setStatus(s)
      connectWs()
      startFrameStream()
      loadLogs()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed')
    } finally {
      setConnecting(false)
    }
  }

  const handlePresetConnect = async (preset: CameraPreset) => {
    setError('')
    setConnecting(true)
    setRtspUrl(`dahua://${preset.host}`)
    setLocation(preset.location)
    try {
      const s = await api.connectStreamPreset(preset.id, Array.from(activeTargets))
      setConnected(true)
      setStatus(s)
      connectWs()
      startFrameStream()
      loadLogs()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Connection failed')
    } finally {
      setConnecting(false)
    }
  }

  const handleDisconnect = async () => {
    wsRef.current?.close()
    stopFrameStream()
    await api.disconnectStream()
    setConnected(false)
    setStatus(null)
  }

  const toggleTarget = async (name: string) => {
    const next = new Set(activeTargets)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    setActiveTargets(next)
    if (connected) {
      const s = await api.setTargets(Array.from(next))
      setStatus(s)
    }
  }

  const statusColor = (s: string) => {
    if (s === 'MATCH') return 'text-success border-success/50 bg-success/15'
    if (s === 'ALERT') return 'text-critical border-critical/50 bg-critical/15'
    if (s === 'PARTIAL') return 'text-surface-tint border-surface-tint/50 bg-surface-tint/15'
    return 'text-primary border-border bg-surface-container-high'
  }

  return (
    <div className="h-[calc(100vh-48px)] flex flex-col p-4 gap-4 max-w-[1920px] mx-auto w-full">
      <div className="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-4">
        <Panel className="xl:col-span-9 min-h-0">
          <div className="px-4 py-3 border-b border-border bg-surface-container-low flex flex-wrap items-center gap-3 shrink-0">
            <div className="flex-1 min-w-[180px] relative">
              <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary text-base">link</span>
              <input
                className="input-field pl-9"
                placeholder="RTSP URL or webcam:0"
                value={rtspUrl}
                onChange={(e) => setRtspUrl(e.target.value)}
                disabled={connected}
              />
            </div>
            <input
              className="input-field w-52"
              placeholder="Location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              disabled={connected}
            />
            {!connected ? (
              <PrimaryButton onClick={handleConnect} icon="cast_connected" disabled={connecting}>
                {connecting ? 'Connecting…' : 'Connect Feed'}
              </PrimaryButton>
            ) : (
              <DangerButton onClick={handleDisconnect}>Disconnect</DangerButton>
            )}
            {error && <span className="text-critical text-xs font-mono w-full">{error}</span>}
            {!connected && presets.length > 0 && (
              <div className="w-full flex flex-wrap items-center gap-2 pt-1">
                <span className="text-[11px] font-mono text-text-secondary uppercase tracking-wider">Quick connect</span>
                {presets.map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    disabled={connecting}
                    onClick={() => handlePresetConnect(preset)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded border border-surface-tint/40 bg-surface-container hover:bg-surface-container-high hover:border-surface-tint transition-colors disabled:opacity-50"
                  >
                    <span className="material-symbols-outlined text-surface-tint text-base">videocam</span>
                    <span className="text-xs font-medium text-text-primary">{preset.name}</span>
                    <span className="text-[10px] font-mono text-text-secondary">{preset.host}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className={`flex-1 min-h-0 relative bg-black ${connected ? 'border-glow-pulse' : ''}`}>
            {connected && frameUrl ? (
              <img src={frameUrl} alt="Live feed" className="absolute inset-0 w-full h-full object-contain" />
            ) : connected ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-text-secondary">
                <span className="material-symbols-outlined text-5xl mb-3 opacity-40 animate-pulse">videocam</span>
                <p className="text-sm font-mono">Loading stream…</p>
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-text-secondary">
                <span className="material-symbols-outlined text-6xl mb-3 opacity-40">videocam</span>
                <p className="text-sm font-mono">Connect a feed to begin live monitoring</p>
              </div>
            )}
            {connected && (
              <>
                <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/70 backdrop-blur-sm border border-border px-2.5 py-1 rounded">
                  <div className="w-2 h-2 rounded-full bg-critical blinking-dot" />
                  <span className="font-mono text-[11px] text-critical font-semibold tracking-wider">LIVE</span>
                </div>
                <div className="absolute top-3 right-3 bg-black/70 backdrop-blur-sm border border-border px-2.5 py-1 rounded flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-text-secondary text-sm">location_on</span>
                  <span className="font-mono text-xs text-text-primary">{location}</span>
                </div>
                <div className="absolute bottom-3 left-3 font-mono text-xs text-surface-tint bg-black/70 backdrop-blur-sm border border-border px-2.5 py-1 rounded">
                  FPS {status?.fps ?? 0} · det {status?.detect_fps ?? 0} · faces {status?.faces_seen ?? 0} · hit {status?.matches ?? 0} · {status?.display_resolution || status?.resolution || '—'} · drop {status?.dropped_frames ?? 0} · {status?.ort_provider || (status?.gpu_enabled ? 'GPU' : 'CPU')}
                </div>
              </>
            )}
          </div>
        </Panel>

        <Panel className="xl:col-span-3 min-h-0">
          <PanelHeader title="Active Targets" count={activeTargets.size} />
          <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-2">
            {persons.map((p) => (
              <div
                key={p.name}
                onClick={() => toggleTarget(p.name)}
                className={`flex items-center gap-3 p-2.5 rounded border cursor-pointer transition-all ${
                  activeTargets.has(p.name)
                    ? 'bg-surface-container border-surface-tint/40'
                    : 'bg-surface-dim border-border opacity-50 hover:opacity-80'
                }`}
              >
                <div className="w-10 h-10 rounded overflow-hidden border border-border shrink-0 bg-surface-container">
                  {p.thumbnail ? (
                    <img src={p.thumbnail} alt={p.name} className="avatar-img" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center">
                      <span className="material-symbols-outlined text-text-secondary text-lg">person</span>
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">{p.name}</div>
                  <div className="text-[11px] font-mono text-text-secondary">ID {p.id}</div>
                </div>
                <div className={`w-8 h-4 rounded-full shrink-0 relative transition-colors ${activeTargets.has(p.name) ? 'bg-surface-tint' : 'bg-surface-container-high'}`}>
                  <div className={`absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full transition-all ${activeTargets.has(p.name) ? 'right-0.5 bg-[#0A0A0A]' : 'left-0.5 bg-text-secondary'}`} />
                </div>
              </div>
            ))}
            {persons.length === 0 && (
              <EmptyState icon="person_add" title="No targets enrolled" subtitle="Add persons in Face Database" />
            )}
          </div>
        </Panel>
      </div>

      <Panel className="h-72 shrink-0">
        <PanelHeader
          title="Detection Logs"
          count={logs.length}
          action={
            <SecondaryButton onClick={() => api.exportLogs('livestream')} icon="download">
              Export .xlsx
            </SecondaryButton>
          }
        />
        <div className="flex-1 overflow-y-auto custom-scrollbar p-3">
          {logs.length === 0 ? (
            <EmptyState icon="history" title="No detections yet" subtitle="Events appear here when faces are recognized on the live feed" />
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
              {logs.map((log) => (
                <div key={log.id} className="bg-surface-dim border border-border rounded p-2.5 hover:border-surface-tint/50 transition-colors">
                  <div className="aspect-video rounded overflow-hidden border border-border relative bg-black mb-2">
                    {log.snapshot_url ? (
                      <img src={log.snapshot_url} alt="" className="w-full h-full object-cover object-top" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <span className="material-symbols-outlined text-text-secondary">person</span>
                      </div>
                    )}
                    <div className={`absolute top-1 right-1 border px-1.5 py-0.5 rounded text-[9px] font-bold font-mono ${statusColor(log.status)}`}>
                      {log.status}
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-semibold text-text-primary truncate">{log.name}</span>
                    <span className="text-xs font-mono text-surface-tint">{(log.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="text-[10px] font-mono text-text-secondary mt-1 truncate">{log.timestamp} · {log.location}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Panel>
    </div>
  )
}
