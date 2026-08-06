export interface Person {
  name: string
  id: string
  is_unknown: boolean
  image_count: number
  embedding_count: number
  thumbnail: string | null
}

export interface DetectionLog {
  id: string
  name: string
  score: number
  status: string
  location: string
  timestamp: string
  snapshot_url: string | null
  source?: string
}

export interface StreamStatus {
  connected: boolean
  source: string
  fps: number
  detect_fps?: number
  resolution: string
  display_resolution?: string
  dropped_frames?: number
  faces_seen?: number
  matches?: number
  location: string
  active_targets: string[]
  identity_count: number
  frame_seq?: number
  gpu_enabled?: boolean
  ort_provider?: string
}

export interface CameraPreset {
  id: string
  name: string
  brand: string
  host: string
  location: string
  channel: number
  subtype: number
}

export interface VideoJob {
  id: string
  status: string
  progress: number
  total_frames: number
  current_frame: number
  filename?: string
  output_url?: string | null
  detections: Array<{
    name: string
    score: number
    timestamp: string
    frame: number
    start_sec: number
  }>
  error?: string
  targets?: string[]
  can_rescan?: boolean
}

const BASE = ''
const REQUEST_TIMEOUT_MS = 20_000

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    const res = await fetch(`${BASE}${path}`, {
      ...options,
      signal: options?.signal ?? controller.signal,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || res.statusText)
    }
    return res.json()
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('Backend request timed out. Check that the API is running on port 8000 and the camera source is available.')
    }
    throw error
  } finally {
    window.clearTimeout(timeout)
  }
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),

  listPersons: (search = '') =>
    request<Person[]>(`/api/persons${search ? `?search=${encodeURIComponent(search)}` : ''}`),

  createPerson: async (name: string, image: File) => {
    const form = new FormData()
    form.append('name', name)
    form.append('image', image)
    return request<{ name: string }>('/api/persons', { method: 'POST', body: form })
  },

  deletePerson: (name: string) =>
    request<{ deleted: string }>(`/api/persons/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  listStreamPresets: () => request<CameraPreset[]>('/api/stream/presets'),

  connectStream: (source: string, location: string, targets?: string[]) =>
    request<StreamStatus>('/api/stream/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, location, targets }),
    }),

  connectStreamPreset: (presetId: string, targets?: string[]) =>
    request<StreamStatus>('/api/stream/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset_id: presetId, targets }),
    }),

  disconnectStream: () =>
    request<{ connected: boolean }>('/api/stream/disconnect', { method: 'POST' }),

  streamStatus: () => request<StreamStatus>('/api/stream/status'),

  setTargets: (targets: string[]) =>
    request<StreamStatus>('/api/stream/targets', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(targets),
    }),

  getLogs: (limit = 50, source?: string) =>
    request<DetectionLog[]>(`/api/logs?limit=${limit}${source ? `&source=${source}` : ''}`),

  clearLogs: (source?: string) =>
    request<{ deleted: number; deleted_snapshots: number }>(
      `/api/logs${source ? `?source=${encodeURIComponent(source)}` : ''}`,
      { method: 'DELETE' },
    ),

  exportLogs: (source?: string) => {
    window.open(`/api/logs/export${source ? `?source=${source}` : ''}`, '_blank')
  },

  uploadVideo: async (file: File, targets: string[]) => {
    const form = new FormData()
    form.append('video', file)
    form.append('targets', targets.join(','))
    return request<{ job_id: string; filename: string }>('/api/video/upload', { method: 'POST', body: form })
  },

  listVideoJobs: () => request<VideoJob[]>('/api/video/jobs'),

  getVideoJob: (jobId: string) => request<VideoJob>(`/api/video/jobs/${jobId}`),

  deleteVideoJob: (jobId: string) =>
    request<{ deleted: string }>(`/api/video/jobs/${jobId}`, { method: 'DELETE' }),

  rescanVideoJob: (jobId: string, targets: string[]) =>
    request<VideoJob>(`/api/video/jobs/${jobId}/rescan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ targets }),
    }),

  fetchStreamSnapshot: async () => {
    const res = await fetch(`/api/stream/snapshot?t=${Date.now()}`, { cache: 'no-store' })
    if (!res.ok) throw new Error('Snapshot failed')
    const seq = parseInt(res.headers.get('X-Frame-Seq') || '0', 10)
    const blob = await res.blob()
    return { blob, seq }
  },

  mjpegUrl: () => `/api/stream/mjpeg?t=${Date.now()}`,
}
