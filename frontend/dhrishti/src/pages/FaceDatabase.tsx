import { useCallback, useEffect, useState } from 'react'
import { api, Person } from '../api/client'
import { EmptyState, Panel, PrimaryButton, SecondaryButton, IconButton, GhostButton } from '../components/ui'

export default function FaceDatabase() {
  const [persons, setPersons] = useState<Person[]>([])
  const [search, setSearch] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [newName, setNewName] = useState('')
  const [newImage, setNewImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    const data = await api.listPersons(search)
    setPersons(data.filter((p) => !p.is_unknown))
  }, [search])

  useEffect(() => { load() }, [load])

  const handleImageSelect = (file: File | null) => {
    setNewImage(file)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(file ? URL.createObjectURL(file) : null)
  }

  const handleAdd = async () => {
    if (!newName.trim() || !newImage) return
    setLoading(true)
    setError('')
    try {
      await api.createPerson(newName.trim(), newImage)
      setShowModal(false)
      setNewName('')
      handleImageSelect(null)
      load()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete ${name} from database?`)) return
    await api.deletePerson(name)
    load()
  }

  return (
    <div className="h-[calc(100vh-48px)] flex flex-col p-4 gap-4 max-w-[1920px] mx-auto w-full">
      <Panel className="shrink-0">
        <div className="px-6 py-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-xl font-semibold text-text-primary">Database Management</h1>
            <p className="text-sm text-text-secondary mt-1">
              {persons.length} enrolled {persons.length === 1 ? 'identity' : 'identities'} · System active
            </p>
          </div>
          <div className="flex items-center gap-3 w-full md:w-auto">
            <div className="relative flex-1 md:w-64">
              <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary text-base">search</span>
              <input
                className="input-field pl-9"
                placeholder="Search name or ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <PrimaryButton onClick={() => setShowModal(true)} icon="add">
              Add Person
            </PrimaryButton>
          </div>
        </div>
      </Panel>

      <Panel className="flex-1 min-h-0">
        <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
          {persons.length === 0 ? (
            <EmptyState icon="groups" title="No persons enrolled" subtitle="Add your first identity to enable face recognition" />
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
              {persons.map((p) => (
                <div
                  key={p.name}
                  className="bg-surface-dim border border-border rounded-lg overflow-hidden group hover:border-surface-tint/40 transition-all"
                >
                  <div className="aspect-[3/4] relative overflow-hidden bg-black">
                    {p.thumbnail ? (
                      <img src={p.thumbnail} alt={p.name} className="avatar-img" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <span className="material-symbols-outlined text-4xl text-text-secondary">person</span>
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <IconButton icon="delete" variant="danger" onClick={() => handleDelete(p.name)} />
                    </div>
                  </div>
                  <div className="p-3 border-t border-border">
                    <h3 className="text-sm font-semibold text-text-primary truncate">{p.name}</h3>
                    <p className="text-[11px] font-mono text-text-secondary mt-0.5">ID {p.id} · {p.embedding_count} views</p>
                  </div>
                </div>
              ))}

              <div className="bg-surface-dim border border-border rounded-lg overflow-hidden flex flex-col">
                <div className="aspect-[3/4] p-3 flex items-center justify-center">
                  <GhostButton onClick={() => setShowModal(true)} icon="person_add" className="!flex-col !gap-2 !py-6">
                    Add New
                  </GhostButton>
                </div>
                <div className="p-3 border-t border-border">
                  <h3 className="text-sm text-text-secondary italic">New Entry</h3>
                  <p className="text-[11px] font-mono text-text-secondary/60 mt-0.5">ID pending</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </Panel>

      {showModal && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 backdrop-blur-sm">
          <div className="bg-surface border border-border rounded-lg w-full max-w-md overflow-hidden">
            <div className="px-6 py-4 border-b border-border">
              <h2 className="text-lg font-semibold text-text-primary">Register New Person</h2>
              <p className="text-sm text-text-secondary mt-1">Upload a clear, front-facing photo for best results</p>
            </div>
            <div className="p-6 space-y-4">
              <input
                className="input-field"
                placeholder="Full name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
              />
              <div
                onClick={() => document.getElementById('face-upload')?.click()}
                className="border-2 border-dashed border-border rounded-lg p-4 cursor-pointer hover:border-surface-tint/60 transition-colors flex flex-col items-center"
              >
                {preview ? (
                  <img src={preview} alt="Preview" className="w-32 h-40 object-cover object-top rounded border border-border" />
                ) : (
                  <>
                    <span className="material-symbols-outlined text-3xl text-text-secondary mb-2">add_a_photo</span>
                    <span className="text-sm text-text-secondary">Click to select photo</span>
                  </>
                )}
                <input
                  id="face-upload"
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => handleImageSelect(e.target.files?.[0] || null)}
                />
              </div>
              {error && <p className="text-critical text-sm">{error}</p>}
            </div>
            <div className="px-6 py-4 border-t border-border flex gap-3 justify-end bg-surface-container-low">
              <SecondaryButton onClick={() => { setShowModal(false); handleImageSelect(null) }}>
                Cancel
              </SecondaryButton>
              <PrimaryButton onClick={handleAdd} disabled={loading || !newName.trim() || !newImage}>
                {loading ? 'Registering...' : 'Register'}
              </PrimaryButton>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
