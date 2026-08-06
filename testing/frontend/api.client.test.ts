import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api } from '@app/api/client'

describe('api client', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('health calls /api/health', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'ok' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const data = await api.health()
    expect(data.status).toBe('ok')
    expect(fetchMock).toHaveBeenCalledWith('/api/health', undefined)
  })

  it('listPersons encodes search', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    })
    vi.stubGlobal('fetch', fetchMock)
    await api.listPersons('Ur vi')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/persons?search=Ur%20vi')
  })

  it('throws on non-ok response', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'Select targets' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    await expect(api.listPersons()).rejects.toThrow('Select targets')
  })

  it('setTargets sends JSON body', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ connected: false, active_targets: ['Alice'] }),
    })
    vi.stubGlobal('fetch', fetchMock)
    await api.setTargets(['Alice'])
    const [, options] = fetchMock.mock.calls[0]
    expect(options.method).toBe('PUT')
    expect(options.body).toBe(JSON.stringify(['Alice']))
  })

  it('deleteVideoJob uses DELETE', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ deleted: 'abc' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const out = await api.deleteVideoJob('abc')
    expect(out.deleted).toBe('abc')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/video/jobs/abc')
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE')
  })

  it('mjpegUrl includes cache buster', () => {
    const url = api.mjpegUrl()
    expect(url.startsWith('/api/stream/mjpeg?t=')).toBe(true)
  })
})
