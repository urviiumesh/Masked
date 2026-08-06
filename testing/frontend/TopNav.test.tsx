import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import TopNav from '@app/components/TopNav'

describe('TopNav', () => {
  it('renders brand and navigation links', () => {
    render(
      <MemoryRouter>
        <TopNav />
      </MemoryRouter>,
    )
    expect(screen.getByText('DHRISHTI')).toBeInTheDocument()
    expect(screen.getByText('Livestream')).toBeInTheDocument()
    expect(screen.getByText('Video Processing')).toBeInTheDocument()
    expect(screen.getByText('Face Database')).toBeInTheDocument()
  })

  it('marks livestream active on home route', () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <TopNav />
      </MemoryRouter>,
    )
    const live = screen.getByText('Livestream')
    expect(live.className).toContain('text-primary')
  })
})
