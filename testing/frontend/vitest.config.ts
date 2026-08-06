import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const frontendSrc = path.resolve(__dirname, '../../frontend/dhrishti/src')
const nm = path.resolve(__dirname, 'node_modules')

export default defineConfig({
  plugins: [react()],
  root: __dirname,
  resolve: {
    alias: {
      '@app': frontendSrc,
      react: path.join(nm, 'react'),
      'react-dom': path.join(nm, 'react-dom'),
      'react-router': path.join(nm, 'react-router'),
      'react-router-dom': path.join(nm, 'react-router-dom'),
    },
    dedupe: ['react', 'react-dom', 'react-router', 'react-router-dom'],
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./setup.ts'],
    include: ['./**/*.{test,spec}.{ts,tsx}'],
  },
})
