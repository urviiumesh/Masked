import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '@app/App'

vi.mock('@app/pages/Livestream', () => ({
  default: () => <div>Livestream Page</div>,
}))
vi.mock('@app/pages/VideoProcessing', () => ({
  default: () => <div>Video Page</div>,
}))
vi.mock('@app/pages/FaceDatabase', () => ({
  default: () => <div>Database Page</div>,
}))

describe('App', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders livestream route content', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <App />
      </MemoryRouter>,
    )
    expect(screen.getByText('Livestream Page')).toBeInTheDocument()
    expect(screen.getByText('DHRISHTI')).toBeInTheDocument()
  })

  it('renders video route content', () => {
    render(
      <MemoryRouter initialEntries={['/video']}>
        <App />
      </MemoryRouter>,
    )
    expect(screen.getByText('Video Page')).toBeInTheDocument()
  })

  it('renders database route content', () => {
    render(
      <MemoryRouter initialEntries={['/database']}>
        <App />
      </MemoryRouter>,
    )
    expect(screen.getByText('Database Page')).toBeInTheDocument()
  })
})
