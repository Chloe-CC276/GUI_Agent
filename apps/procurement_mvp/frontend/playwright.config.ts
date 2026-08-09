import { defineConfig, devices } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendRoot = path.dirname(fileURLToPath(import.meta.url))
const mvpRoot = path.resolve(frontendRoot, '..')
const backendRoot = path.join(mvpRoot, 'backend')
const python = path.join(mvpRoot, '.venv', 'Scripts', 'python.exe')

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5174',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: `"${python}" -m uvicorn app.main:app --host 127.0.0.1 --port 8010`,
      cwd: backendRoot,
      env: {
        ...process.env,
        DATABASE_URL: 'sqlite:///./e2e_s0.db',
      },
      url: 'http://127.0.0.1:8010/health',
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5174 --strictPort',
      cwd: frontendRoot,
      env: {
        ...process.env,
        VITE_API_PROXY_TARGET: 'http://127.0.0.1:8010',
      },
      url: 'http://127.0.0.1:5174',
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
})
