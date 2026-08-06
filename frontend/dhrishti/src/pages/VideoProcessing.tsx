import { useCallback, useEffect, useRef, useState } from 'react'
import { api, Person, VideoJob } from '../api/client'
import { EmptyState, Panel, PanelHeader, PrimaryButton, SecondaryButton, DangerButton } from '../components/ui'

type QueueItem = {
  localId: string
  fileName: string
  file?: File
  phase: 'queued' | 'uploading' | 'uploaded' | 'processing' | 'completed' | 'failed'
  progress: number
  jobId?: string
  error?: string
  outputUrl?: string
}

const SELECTED_KEY = 'dhrishti.video.selected'
const ACTIVE_KEY = 'dhrishti.video.activeJobId'

export default function VideoProcessing() {
  const [persons, setPersons] = useState<Person[]>([])
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<Set<string>>(() => {
    try {
      const raw = sessionStorage.getItem(SELECTED_KEY)
      return raw ? new Set(JSON.parse(raw) as string[]) : new Set()
    } catch {
      return new Set()
    }
  })
  const [queue, setQueue] = useState<QueueItem[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [activeJob, setActiveJob] = useState<VideoJob | null>(null)
  const [videoKey, setVideoKey] = useState(0)
  const [videoError, setVideoError] = useState('')
  const [error, setError] = useState('')
  const [restored, setRestored] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const queueRef = useRef<QueueItem[]>([])
  const processingRef = useRef(false)
  const selectedRef = useRef<Set<string>>(selected)

  const loadPersons = useCallback(async () => {
    const data = await api.listPersons()
    setPersons(data.filter((p) => !p.is_unknown))
  }, [])

  const updateItem = useCallback((localId: string, patch: Partial<QueueItem>) => {
    setQueue((prev) => {
      const next = prev.map((item) => (item.localId === localId ? { ...item, ...patch } : item))
      queueRef.current = next
      return next
    })
  }, [])

  const showJob = useCallback((job: VideoJob, localId?: string) => {
    setActiveJob(job)
    if (localId) setActiveId(localId)
    sessionStorage.setItem(ACTIVE_KEY, job.id)
    if (job.status === 'completed' && job.output_url) {
      setVideoKey(Date.now())
      setVideoError('')
    }
  }, [])

  const pollJob = useCallback((localId: string, jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current)
    let tick = 0
    pollRef.current = setInterval(async () => {
      try {
        const j = await api.getVideoJob(jobId)
        const phase = j.status === 'completed' ? 'completed' : j.status === 'failed' ? 'failed' : 'processing'
        tick += 1
        updateItem(localId, {
          progress: j.progress,
          phase,
          error: j.error,
          outputUrl: j.output_url || undefined,
          jobId: j.id,
          fileName: j.filename || queueRef.current.find((q) => q.localId === localId)?.fileName || 'video',
        })
        const done = j.status === 'completed' || j.status === 'failed'
        const shouldRefreshUi = done || tick % 2 === 0
        if (shouldRefreshUi) {
          setActiveId((cur) => {
            if (cur === localId || cur === null) {
              if (phase === 'processing') {
                setActiveJob((prev) => ({
                  ...j,
                  detections: prev?.id === j.id ? (prev.detections || []) : (j.detections || []).slice(-30),
                }))
              } else {
                showJob(j, localId)
              }
            }
            return cur ?? localId
          })
        }
        if (done) {
          if (pollRef.current) clearInterval(pollRef.current)
          pollRef.current = null
          processingRef.current = false
          if (j.status === 'completed') showJob(j, localId)
          const next = queueRef.current.find((item) => item.phase === 'queued' && item.file)
          if (next) setTimeout(() => startNextRef.current(), 300)
        }
      } catch (e) {
        updateItem(localId, { phase: 'failed', error: e instanceof Error ? e.message : 'Polling failed' })
        if (pollRef.current) clearInterval(pollRef.current)
        pollRef.current = null
        processingRef.current = false
      }
    }, 2500)
  }, [showJob, updateItem])

  useEffect(() => {
    loadPersons().catch((e) => setError(e instanceof Error ? e.message : 'Could not load persons'))
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [loadPersons])

  useEffect(() => {
    queueRef.current = queue
  }, [queue])

  useEffect(() => {
    selectedRef.current = selected
    sessionStorage.setItem(SELECTED_KEY, JSON.stringify(Array.from(selected)))
  }, [selected])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const jobs = await api.listVideoJobs()
        if (cancelled) return
        const items: QueueItem[] = jobs.map((j) => ({
          localId: `job-${j.id}`,
          fileName: j.filename || `job-${j.id}`,
          phase: j.status === 'completed' ? 'completed' : j.status === 'failed' ? 'failed' : j.status === 'queued' ? 'queued' : 'processing',
          progress: j.progress,
          jobId: j.id,
          error: j.error,
          outputUrl: j.output_url || undefined,
        }))
        setQueue((prev) => {
          const byJob = new Map(items.map((i) => [i.jobId!, i]))
          const keptLocal = prev.filter((p) => p.phase === 'queued' && p.file && !p.jobId)
          const merged = [...keptLocal, ...items]
          queueRef.current = merged
          return merged
        })
        const preferredId = sessionStorage.getItem(ACTIVE_KEY)
        const preferred = jobs.find((j) => j.id === preferredId) || jobs.find((j) => j.status === 'processing') || jobs.find((j) => j.status === 'completed')
        if (preferred) {
          showJob(preferred, `job-${preferred.id}`)
          if (preferred.status === 'processing' || preferred.status === 'queued') {
            processingRef.current = true
            pollJob(`job-${preferred.id}`, preferred.id)
          }
        }
      } catch {
        /* ignore */
      } finally {
        if (!cancelled) setRestored(true)
      }
    })()
    return () => { cancelled = true }
  }, [pollJob, showJob])

  const filtered = persons.filter(
    (p) => !search || p.name.toLowerCase().includes(search.toLowerCase())
  )

  const toggleSelect = (name: string) => {
    const next = new Set(selected)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    setSelected(next)
  }

  const runUpload = useCallback(async (item: QueueItem) => {
    if (processingRef.current) return
    if (!item.file) {
      setError('Re-add this video file to process it')
      return
    }
    const targets = Array.from(selectedRef.current)
    if (targets.length === 0) {
      setError('Select at least one target before processing')
      updateItem(item.localId, { phase: 'queued' })
      return
    }
    processingRef.current = true
    setActiveId(item.localId)
    setActiveJob(null)
    setVideoError('')
    updateItem(item.localId, { phase: 'uploading', progress: 0, error: undefined })
    try {
      const { job_id, filename } = await api.uploadVideo(item.file, targets)
      updateItem(item.localId, { phase: 'processing', jobId: job_id, progress: 0, fileName: filename || item.fileName })
      const j = await api.getVideoJob(job_id)
      showJob({ ...j, filename: j.filename || filename }, item.localId)
      pollJob(item.localId, job_id)
    } catch (e) {
      updateItem(item.localId, { phase: 'failed', error: e instanceof Error ? e.message : 'Upload failed' })
      processingRef.current = false
      const next = queueRef.current.find((q) => q.phase === 'queued' && q.file)
      if (next) setTimeout(() => startNextRef.current(), 300)
    }
  }, [pollJob, showJob, updateItem])

  const startNextRef = useRef<() => void>(() => {})
  startNextRef.current = () => {
    if (processingRef.current) return
    if (selectedRef.current.size === 0) {
      setError('Select at least one target before processing')
      return
    }
    const next = queueRef.current.find((item) => item.phase === 'queued' && item.file)
    if (next) runUpload(next)
  }

  const addFiles = (files: FileList | File[]) => {
    const videos = Array.from(files).filter((f) => f.type.startsWith('video/') || /\.(mp4|mkv|avi|mov|webm)$/i.test(f.name))
    if (videos.length === 0) return
    const items: QueueItem[] = videos.map((file) => ({
      localId: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      fileName: file.name,
      file,
      phase: 'queued',
      progress: 0,
    }))
    setQueue((prev) => {
      const next = [...prev, ...items]
      queueRef.current = next
      return next
    })
    setError('')
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files)
  }

  const handleProcess = () => {
    if (selectedRef.current.size === 0) {
      setError('Select at least one target before processing')
      return
    }
    const pending = queueRef.current.filter((item) => item.phase === 'queued' && item.file)
    if (pending.length === 0) {
      setError('Add one or more video files first')
      return
    }
    setError('')
    startNextRef.current()
  }

  const removeItem = (localId: string) => {
    setQueue((prev) => {
      const next = prev.filter((item) => item.localId !== localId)
      queueRef.current = next
      return next
    })
    if (activeId === localId) {
      if (pollRef.current) clearInterval(pollRef.current)
      pollRef.current = null
      setActiveId(null)
      setActiveJob(null)
      sessionStorage.removeItem(ACTIVE_KEY)
    }
  }

  const handleDelete = async (item: QueueItem, e?: React.MouseEvent) => {
    e?.stopPropagation()
    if (item.phase === 'uploading' || item.phase === 'processing') {
      setError('Wait for processing to finish before deleting')
      return
    }
    if (!item.jobId) {
      removeItem(item.localId)
      return
    }
    try {
      await api.deleteVideoJob(item.jobId)
      removeItem(item.localId)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    }
  }

  const handleRescan = async () => {
    const item = queueRef.current.find((q) => q.localId === activeId)
    const jobId = item?.jobId || activeJob?.id
    if (!jobId) {
      setError('Select a processed video to rescan')
      return
    }
    const targets = Array.from(selectedRef.current)
    if (targets.length === 0) {
      setError('Select at least one target before rescanning')
      return
    }
    if (isBusy) {
      setError('Wait for the current job to finish')
      return
    }
    setError('')
    setVideoError('')
    processingRef.current = true
    const localId = item?.localId || `job-${jobId}`
    setActiveId(localId)
    updateItem(localId, {
      phase: 'processing',
      progress: 0,
      error: undefined,
      outputUrl: undefined,
      jobId,
    })
    try {
      const j = await api.rescanVideoJob(jobId, targets)
      showJob(j, localId)
      pollJob(localId, jobId)
    } catch (err) {
      processingRef.current = false
      updateItem(localId, {
        phase: 'failed',
        error: err instanceof Error ? err.message : 'Rescan failed',
      })
      setError(err instanceof Error ? err.message : 'Rescan failed')
    }
  }

  const loadJobForItem = async (item: QueueItem) => {
    if (!item.jobId) {
      setActiveId(item.localId)
      return
    }
    setActiveId(item.localId)
    try {
      const j = await api.getVideoJob(item.jobId)
      showJob(j, item.localId)
      if (j.status === 'processing' || j.status === 'queued') {
        processingRef.current = true
        pollJob(item.localId, item.jobId)
      }
    } catch {
      setError('Could not load job')
    }
  }

  const phaseIcon = (phase: QueueItem['phase']) => {
    if (phase === 'completed') return 'check_circle'
    if (phase === 'failed') return 'error'
    if (phase === 'uploading') return 'cloud_upload'
    if (phase === 'uploaded') return 'done'
    if (phase === 'processing') return 'hourglass_top'
    return 'schedule'
  }

  const phaseColor = (phase: QueueItem['phase']) => {
    if (phase === 'completed') return 'text-success'
    if (phase === 'failed') return 'text-critical'
    if (phase === 'uploaded') return 'text-surface-tint'
    if (phase === 'uploading' || phase === 'processing') return 'text-surface-tint'
    return 'text-text-secondary'
  }

  const isBusy = queue.some((q) => ['uploading', 'uploaded', 'processing'].includes(q.phase))
  const outputUrl = activeJob?.status === 'completed' ? (activeJob.output_url || `/api/video/jobs/${activeJob.id}/output`) : null
  const outputSrc = outputUrl ? `${outputUrl}?v=${videoKey}` : null

  return (
    <div className="h-[calc(100vh-48px)] flex flex-col p-4 gap-4 max-w-[1920px] mx-auto w-full">
      <div className="flex-1 min-h-0 grid grid-cols-1 xl:grid-cols-12 gap-4">
        <Panel className="xl:col-span-3 min-h-0">
          <PanelHeader title="Input Source" count={queue.length || undefined} />
          <div className="p-4 flex flex-col gap-4 flex-1 min-h-0">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              className="border-2 border-dashed border-border rounded-lg flex flex-col items-center justify-center p-6 bg-surface-dim cursor-pointer hover:border-surface-tint/60 transition-colors min-h-[120px] group"
            >
              <span className="material-symbols-outlined text-3xl text-text-secondary group-hover:text-surface-tint transition-colors mb-2">video_file</span>
              <span className="text-sm text-text-primary text-center font-medium">Drop videos or click to add</span>
              <span className="text-xs font-mono text-text-secondary mt-1">Select targets, then Process · state kept across tabs</span>
              <input ref={fileRef} type="file" accept="video/*" multiple className="hidden" onChange={(e) => { e.target.files && addFiles(e.target.files); e.target.value = '' }} />
            </div>

            {queue.length > 0 && (
              <div className="flex flex-col gap-1.5 max-h-40 overflow-y-auto custom-scrollbar">
                {queue.map((item) => (
                  <div
                    key={item.localId}
                    onClick={() => loadJobForItem(item)}
                    className={`flex items-center gap-2 p-2 rounded border text-xs cursor-pointer transition-colors ${
                      activeId === item.localId ? 'border-surface-tint/50 bg-surface-container' : 'border-border bg-surface-dim hover:border-border/80'
                    }`}
                  >
                    <span className={`material-symbols-outlined text-base shrink-0 ${phaseColor(item.phase)}`}>{phaseIcon(item.phase)}</span>
                    <div className="flex-1 min-w-0">
                      <div className="truncate text-text-primary font-medium">{item.fileName}</div>
                      <div className="font-mono text-[10px] text-text-secondary">
                        {item.phase === 'queued' && (item.file ? 'Queued' : 'Ready to rescan/delete')}
                        {item.phase === 'uploading' && 'Uploading…'}
                        {item.phase === 'uploaded' && 'Upload complete — starting…'}
                        {item.phase === 'processing' && `Processing ${item.progress.toFixed(0)}%`}
                        {item.phase === 'completed' && 'Done — click to preview'}
                        {item.phase === 'failed' && (item.error || 'Failed')}
                      </div>
                    </div>
                    {!['uploading', 'processing'].includes(item.phase) && (
                      <button
                        type="button"
                        title="Delete video"
                        onClick={(e) => handleDelete(item, e)}
                        className="text-text-secondary hover:text-critical shrink-0"
                      >
                        <span className="material-symbols-outlined text-sm">delete</span>
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="flex-1 min-h-0 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold text-text-primary uppercase tracking-widest">Targets</span>
                <span className="text-[10px] font-mono bg-surface-container-high px-2 py-0.5 rounded border border-border text-text-primary">{selected.size} selected</span>
              </div>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary text-base">search</span>
                <input className="input-field pl-9" placeholder="Search..." value={search} onChange={(e) => setSearch(e.target.value)} />
              </div>
              <div className="flex-1 overflow-y-auto custom-scrollbar space-y-1.5 min-h-0">
                {filtered.map((p) => (
                  <div
                    key={p.name}
                    onClick={() => toggleSelect(p.name)}
                    className={`flex items-center gap-3 p-2 rounded border cursor-pointer transition-all ${
                      selected.has(p.name) ? 'bg-surface-container border-surface-tint/40' : 'bg-surface-dim border-border hover:border-border/80'
                    }`}
                  >
                    <input type="checkbox" checked={selected.has(p.name)} readOnly className="accent-[#E4F222] shrink-0" />
                    <div className="w-9 h-9 rounded overflow-hidden border border-border shrink-0">
                      {p.thumbnail ? (
                        <img src={p.thumbnail} alt="" className="avatar-img" />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-surface-container">
                          <span className="material-symbols-outlined text-text-secondary text-sm">person</span>
                        </div>
                      )}
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm text-text-primary font-medium truncate">{p.name}</div>
                      <div className="text-[10px] font-mono text-text-secondary">ID {p.id}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {error && <p className="text-critical text-xs font-mono">{error}</p>}
            <div className="flex flex-col gap-2">
              <PrimaryButton onClick={handleProcess} disabled={isBusy || !queue.some((q) => q.phase === 'queued' && q.file) || selected.size === 0} icon="play_arrow" className="w-full">
                {isBusy ? 'Processing…' : selected.size === 0 ? 'Select targets first' : queue.some((q) => q.phase === 'queued' && q.file) ? `Process ${queue.filter((q) => q.phase === 'queued' && q.file).length} for ${selected.size} target(s)` : 'Add videos to process'}
              </PrimaryButton>
              <SecondaryButton
                onClick={handleRescan}
                disabled={isBusy || selected.size === 0 || !(activeJob?.can_rescan || (activeJob && ['completed', 'failed'].includes(activeJob.status)))}
                icon="replay"
                className="w-full"
              >
                {selected.size === 0 ? 'Select targets to rescan' : `Rescan with ${selected.size} target(s)`}
              </SecondaryButton>
            </div>
          </div>
        </Panel>

        <div className="xl:col-span-9 flex flex-col gap-4 min-h-0">
          <Panel className="flex-1 min-h-0">
            <PanelHeader
              title="Video Output"
              count={activeJob?.status === 'completed' ? 'Ready' : isBusy ? `${activeJob?.progress ?? 0}%` : restored ? '—' : '…'}
              action={
                <div className="flex items-center gap-2">
                  {activeJob && ['completed', 'failed'].includes(activeJob.status) && (
                    <DangerButton
                      onClick={() => {
                        const item = queue.find((q) => q.jobId === activeJob.id || q.localId === activeId)
                        if (item) handleDelete(item)
                      }}
                    >
                      Delete
                    </DangerButton>
                  )}
                  {outputSrc ? (
                    <SecondaryButton onClick={() => window.open(outputSrc, '_blank')} icon="download">
                      Open / Download
                    </SecondaryButton>
                  ) : null}
                </div>
              }
            />
            <div className="flex-1 min-h-0 relative bg-black">
              {outputSrc && activeJob?.status === 'completed' && !videoError ? (
                <video
                  key={videoKey}
                  src={outputSrc}
                  controls
                  autoPlay
                  playsInline
                  className="absolute inset-0 w-full h-full object-contain"
                  onError={() => setVideoError('Browser could not play this file. Use Open / Download.')}
                />
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-text-secondary gap-3 px-4">
                  <span className="material-symbols-outlined text-5xl opacity-40">videocam</span>
                  <p className="text-sm font-mono text-center">
                    {videoError
                      ? videoError
                      : isBusy
                        ? `Processing ${activeJob?.filename || 'video'} — frame ${activeJob?.current_frame ?? 0} / ${activeJob?.total_frames ?? '?'} (${(activeJob?.progress ?? 0).toFixed(0)}%)`
                        : activeJob?.status === 'completed'
                          ? 'Output ready — use Open / Download if preview is blank'
                          : 'Select targets, add videos, then Process. Progress stays when you switch tabs.'}
                  </p>
                  {outputSrc && (
                    <a href={outputSrc} target="_blank" rel="noreferrer" className="text-surface-tint text-xs font-mono underline">
                      Open processed video
                    </a>
                  )}
                </div>
              )}
              {isBusy && (
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-surface-container-high">
                  <div className="h-full bg-surface-tint transition-all duration-300" style={{ width: `${activeJob?.progress ?? 0}%` }} />
                </div>
              )}
            </div>
          </Panel>

          <Panel className="h-56 shrink-0">
            <PanelHeader
              title="Detection Instances"
              count={activeJob?.detections?.length ?? 0}
              action={
                <SecondaryButton onClick={() => api.exportLogs('video')} icon="download">
                  Export .xlsx
                </SecondaryButton>
              }
            />
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              {(activeJob?.detections ?? []).length === 0 ? (
                <EmptyState icon="frame_inspect" title="No detections recorded" subtitle="Detection instances appear during and after video processing" />
              ) : (
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-container-low border-b border-border">
                    <tr>
                      <th className="text-left px-4 py-2 text-[10px] font-mono text-text-secondary uppercase tracking-wider">Target</th>
                      <th className="text-left px-4 py-2 text-[10px] font-mono text-text-secondary uppercase tracking-wider">Confidence</th>
                      <th className="text-left px-4 py-2 text-[10px] font-mono text-text-secondary uppercase tracking-wider">Timestamp</th>
                      <th className="text-left px-4 py-2 text-[10px] font-mono text-text-secondary uppercase tracking-wider">Frame</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeJob!.detections.map((d, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-surface-container-low transition-colors">
                        <td className="px-4 py-2.5 font-medium text-text-primary">{d.name}</td>
                        <td className="px-4 py-2.5 font-mono text-surface-tint">{(d.score * 100).toFixed(1)}%</td>
                        <td className="px-4 py-2.5 font-mono text-text-secondary text-xs">{d.timestamp}</td>
                        <td className="px-4 py-2.5 font-mono text-text-secondary text-xs">#{d.frame}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  )
}
