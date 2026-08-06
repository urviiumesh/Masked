import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { EmptyState, Panel, PanelHeader } from '@app/components/ui'

describe('ui components', () => {
  it('renders Panel children', () => {
    render(
      <Panel>
        <div>Inside panel</div>
      </Panel>,
    )
    expect(screen.getByText('Inside panel')).toBeInTheDocument()
  })

  it('renders PanelHeader title and count', () => {
    render(<PanelHeader title="Targets" count={3} />)
    expect(screen.getByText('Targets')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('renders EmptyState', () => {
    render(<EmptyState icon="person" title="No faces" subtitle="Add one" />)
    expect(screen.getByText('No faces')).toBeInTheDocument()
    expect(screen.getByText('Add one')).toBeInTheDocument()
  })
})
