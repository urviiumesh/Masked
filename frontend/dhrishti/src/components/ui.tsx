import { ReactNode } from 'react'
import SpecularButton from './SpecularButton'

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`bg-surface border border-border rounded overflow-hidden flex flex-col ${className}`}>
      {children}
    </div>
  )
}

export function PanelHeader({
  title,
  action,
  count,
}: {
  title: string
  action?: ReactNode
  count?: number | string
}) {
  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-surface-container-low shrink-0">
      <div className="flex items-center gap-2">
        <h2 className="text-xs font-mono font-semibold text-text-primary uppercase tracking-widest">{title}</h2>
        {count !== undefined && (
          <span className="bg-surface-container-high text-text-primary text-[10px] font-mono px-2 py-0.5 rounded border border-border">
            {count}
          </span>
        )}
      </div>
      {action}
    </div>
  )
}

function ButtonContent({ icon, children, filled }: { icon?: string; children: ReactNode; filled?: boolean }) {
  return (
    <>
      {icon && (
        <span
          className="material-symbols-outlined text-base"
          style={{ fontVariationSettings: filled ? "'FILL' 1" : "'FILL' 0" }}
        >
          {icon}
        </span>
      )}
      {children}
    </>
  )
}

export function PrimaryButton({
  children,
  onClick,
  disabled,
  className = '',
  icon,
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  className?: string
  icon?: string
}) {
  return (
    <SpecularButton
      size="md"
      radius={4}
      tint="#E4F222"
      tintOpacity={0.85}
      textColor="#0A0A0A"
      lineColor="#f0ff80"
      baseColor="#636900"
      intensity={1.3}
      shineSize={14}
      shineFade={35}
      thickness={1.2}
      proximity={300}
      onClick={onClick}
      disabled={disabled}
      className={`font-semibold ${className}`}
    >
      <ButtonContent icon={icon} filled>{children}</ButtonContent>
    </SpecularButton>
  )
}

export function SecondaryButton({
  children,
  onClick,
  className = '',
  icon,
  disabled,
}: {
  children: ReactNode
  onClick?: () => void
  className?: string
  icon?: string
  disabled?: boolean
}) {
  return (
    <SpecularButton
      size="sm"
      radius={4}
      tint="#1c1b1b"
      tintOpacity={0.9}
      blur={8}
      textColor="#EDEDED"
      lineColor="#c3d000"
      baseColor="#3f3f46"
      intensity={1}
      shineSize={12}
      shineFade={40}
      onClick={onClick}
      disabled={disabled}
      className={className}
    >
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </SpecularButton>
  )
}

export function DangerButton({
  children,
  onClick,
  className = '',
  disabled,
}: {
  children: ReactNode
  onClick?: () => void
  className?: string
  disabled?: boolean
}) {
  return (
    <SpecularButton
      size="md"
      radius={4}
      tint="#1a0a0a"
      tintOpacity={0.7}
      textColor="#EF4444"
      lineColor="#f87171"
      baseColor="#450a0a"
      intensity={1.1}
      shineSize={12}
      onClick={onClick}
      disabled={disabled}
      className={className}
    >
      {children}
    </SpecularButton>
  )
}

export function IconButton({
  icon,
  onClick,
  variant = 'default',
  className = '',
}: {
  icon: string
  onClick?: () => void
  variant?: 'default' | 'danger'
  className?: string
}) {
  const isDanger = variant === 'danger'
  return (
    <SpecularButton
      size="sm"
      radius={999}
      tint={isDanger ? '#1a0a0a' : '#121212'}
      tintOpacity={0.8}
      blur={6}
      textColor={isDanger ? '#EF4444' : '#EDEDED'}
      lineColor={isDanger ? '#f87171' : '#c3d000'}
      baseColor={isDanger ? '#450a0a' : '#3f3f46'}
      intensity={1}
      shineSize={10}
      onClick={onClick}
      className={`!p-2.5 min-w-[2.25rem] min-h-[2.25rem] ${className}`}
    >
      <span className="material-symbols-outlined text-base">{icon}</span>
    </SpecularButton>
  )
}

export function GhostButton({
  children,
  onClick,
  className = '',
  icon,
}: {
  children: ReactNode
  onClick?: () => void
  className?: string
  icon?: string
}) {
  return (
    <SpecularButton
      size="md"
      radius={4}
      tint="#121212"
      tintOpacity={0.5}
      blur={4}
      textColor="#A1A1AA"
      lineColor="#c3d000"
      baseColor="#27272A"
      intensity={0.9}
      shineSize={14}
      onClick={onClick}
      className={`w-full h-full ${className}`}
    >
      <ButtonContent icon={icon}>{children}</ButtonContent>
    </SpecularButton>
  )
}

export function EmptyState({ icon, title, subtitle }: { icon: string; title: string; subtitle?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
      <span className="material-symbols-outlined text-4xl text-text-secondary mb-3">{icon}</span>
      <p className="text-sm text-text-primary font-medium">{title}</p>
      {subtitle && <p className="text-xs text-text-secondary mt-1 font-mono">{subtitle}</p>}
    </div>
  )
}
