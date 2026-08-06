import type { ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import TopNav from './components/TopNav'
import Livestream from './pages/Livestream'
import FaceDatabase from './pages/FaceDatabase'
import VideoProcessing from './pages/VideoProcessing'

function KeepAlivePage({ path, children }: { path: string; children: ReactNode }) {
  const location = useLocation()
  const active = location.pathname === path
  return (
    <div className={active ? 'flex-1 min-h-0 flex flex-col' : 'hidden'}>
      {children}
    </div>
  )
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-[#0A0A0A]">
      <TopNav />
      <KeepAlivePage path="/">
        <Livestream />
      </KeepAlivePage>
      <KeepAlivePage path="/video">
        <VideoProcessing />
      </KeepAlivePage>
      <KeepAlivePage path="/database">
        <FaceDatabase />
      </KeepAlivePage>
    </div>
  )
}
