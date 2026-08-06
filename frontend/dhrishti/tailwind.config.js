/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#131313',
        surface: '#121212',
        'surface-dim': '#131313',
        'surface-container': '#201f1f',
        'surface-container-low': '#1c1b1b',
        'surface-container-high': '#2a2a2a',
        'surface-tint': '#c3d000',
        border: '#27272A',
        'text-primary': '#EDEDED',
        'text-secondary': '#A1A1AA',
        primary: '#ffffff',
        success: '#10B981',
        critical: '#EF4444',
        'primary-fixed-dim': '#c3d000',
        'outline-variant': '#474833',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Geist Mono', 'monospace'],
      },
      spacing: {
        'margin-page': '24px',
        gutter: '16px',
      },
      borderRadius: {
        DEFAULT: '0.125rem',
        lg: '0.25rem',
        xl: '0.5rem',
      },
    },
  },
  plugins: [],
}
