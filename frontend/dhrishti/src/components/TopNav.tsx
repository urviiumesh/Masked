import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Livestream' },
  { to: '/video', label: 'Video Processing' },
  { to: '/database', label: 'Face Database' },
]

export default function TopNav() {
  return (
    <nav className="bg-[#0A0A0A] border-b border-border sticky top-0 z-50 shrink-0">
      <div className="flex items-center h-12 px-6 gap-8 max-w-[1920px] mx-auto">
        <div className="text-sm font-bold tracking-widest text-primary uppercase">DHRISHTI</div>
        <div className="flex items-center gap-6 h-full">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === '/'}
              className={({ isActive }) =>
                `h-full flex items-center text-sm font-medium transition-colors border-b-2 ${
                  isActive
                    ? 'text-primary border-surface-tint'
                    : 'text-text-secondary border-transparent hover:text-primary'
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
