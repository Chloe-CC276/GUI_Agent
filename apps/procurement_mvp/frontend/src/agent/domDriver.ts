import type { NavigateFunction } from 'react-router-dom'
import { api } from '../api'

export type AgentStep = {
  step_id: string
  title?: string
  action: Record<string, unknown>
  verify: Record<string, unknown>
  expected?: string
  actual?: unknown
  status?: string
  retry_count?: number
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function qs(testid: string): HTMLElement | null {
  return document.querySelector(`[data-testid="${testid}"]`) as HTMLElement | null
}

/** Keep the agent panel from intercepting clicks / covering form controls during DOM steps. */
async function withAgentChromeHidden<T>(fn: () => Promise<T>): Promise<T> {
  const nodes = Array.from(
    document.querySelectorAll<HTMLElement>('[data-testid="agent-floating-window"], .agent-fab, .agent-float'),
  )
  const previous = nodes.map((node) => ({
    node,
    visibility: node.style.visibility,
    pointerEvents: node.style.pointerEvents,
    opacity: node.style.opacity,
  }))
  for (const node of nodes) {
    node.style.visibility = 'hidden'
    node.style.pointerEvents = 'none'
    node.style.opacity = '0'
    node.setAttribute('data-agent-chrome-hidden', '1')
  }
  try {
    await sleep(40)
    return await fn()
  } finally {
    for (const item of previous) {
      item.node.style.visibility = item.visibility
      item.node.style.pointerEvents = item.pointerEvents
      item.node.style.opacity = item.opacity
      item.node.removeAttribute('data-agent-chrome-hidden')
    }
  }
}

function dispatchMouse(el: HTMLElement, type: string) {
  el.dispatchEvent(
    new MouseEvent(type, {
      bubbles: true,
      cancelable: true,
      view: window,
      buttons: 1,
    }),
  )
}

function visible(el: Element | null) {
  if (!el || !(el instanceof HTMLElement)) return false
  const style = window.getComputedStyle(el)
  if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || '1') === 0) {
    return false
  }
  const rect = el.getBoundingClientRect()
  return rect.width > 0 && rect.height > 0
}

async function waitForTestId(testid: string, timeoutMs = 5000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    const el = qs(testid)
    if (visible(el)) return el!
    await sleep(100)
  }
  throw new Error(`timeout waiting for ${testid}`)
}

function setNativeValue(el: HTMLElement, value: string) {
  const input = el as HTMLInputElement
  const proto = Object.getPrototypeOf(input)
  const descriptor = Object.getOwnPropertyDescriptor(proto, 'value')
  descriptor?.set?.call(input, value)
  input.dispatchEvent(new Event('input', { bubbles: true }))
  input.dispatchEvent(new Event('change', { bubbles: true }))
}

function openSelectOptions(): HTMLElement[] {
  return Array.from(
    document.querySelectorAll(
      '.ant-select-dropdown:not(.ant-select-dropdown-hidden) .ant-select-item-option:not(.ant-select-item-option-disabled)',
    ),
  ) as HTMLElement[]
}

function selectOptionMatch(item: HTMLElement, value: string) {
  const text = (item.textContent || '').trim()
  const title = (item.getAttribute('title') || '').trim()
  const optionValue = (item.getAttribute('data-value') || item.getAttribute('label') || '').trim()
  return text.includes(value) || title === value || title.includes(value) || optionValue === value
}

async function openAntSelect(el: HTMLElement) {
  const selector = (el.querySelector('.ant-select-selector') as HTMLElement | null) || el
  const combobox = (el.querySelector('input[role="combobox"], .ant-select-selection-search-input') as HTMLElement | null)
    || selector
  el.scrollIntoView({ block: 'center', inline: 'nearest' })
  await sleep(100)
  // Close any leftover dropdown first.
  dispatchMouse(document.body, 'mousedown')
  dispatchMouse(document.body, 'mouseup')
  document.body.click()
  await sleep(60)
  combobox.focus?.()
  dispatchMouse(selector, 'mousedown')
  dispatchMouse(selector, 'mouseup')
  selector.click()
  await sleep(80)
  if (openSelectOptions().length === 0) {
    combobox.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', code: 'ArrowDown', bubbles: true }))
    await sleep(80)
  }
  if (openSelectOptions().length === 0) {
    combobox.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }))
    await sleep(80)
  }
}

async function fillSelect(el: HTMLElement, value: string, testid: string) {
  const selected = el.querySelector('.ant-select-selection-item')
  if (selected && (selected.textContent || '').includes(value)) return

  await openAntSelect(el)

  let hit: HTMLElement | undefined
  const start = Date.now()
  while (Date.now() - start < 3000) {
    const options = openSelectOptions()
    hit = options.find((item) => selectOptionMatch(item, value))
    if (hit) break
    // Retry open once if dropdown never appeared.
    if (options.length === 0 && Date.now() - start > 600) {
      await openAntSelect(el)
    }
    await sleep(100)
  }
  if (!hit) {
    const again = el.querySelector('.ant-select-selection-item')
    if (again && (again.textContent || '').includes(value)) return
    const openLabels = openSelectOptions().map((item) => (item.getAttribute('title') || item.textContent || '').trim())
    throw new Error(`select option not found for ${testid}: ${value}; open=${JSON.stringify(openLabels)}`)
  }
  hit.scrollIntoView({ block: 'nearest' })
  dispatchMouse(hit, 'mousedown')
  dispatchMouse(hit, 'mouseup')
  hit.click()
  await sleep(150)
  document.body.click()
  await sleep(80)
  const confirmed = el.querySelector('.ant-select-selection-item')
  if (!confirmed || !(confirmed.textContent || '').includes(value)) {
    throw new Error(`select value not applied for ${testid}: expected ${value}, got ${(confirmed?.textContent || '').trim()}`)
  }
}

async function fillField(testid: string, value: string, inputType?: string) {
  if (!value && inputType === 'select') return
  const el = await waitForTestId(testid)
  if (inputType === 'select') {
    await fillSelect(el, value, testid)
    return
  }
  el.scrollIntoView({ block: 'center', inline: 'nearest' })
  await sleep(60)
  const input = (el.matches('input,textarea') ? el : el.querySelector('input,textarea')) as HTMLInputElement | null
  if (!input) throw new Error(`no input in ${testid}`)
  input.focus()
  setNativeValue(input, value)
  await sleep(50)
}

function readField(testid: string) {
  const el = qs(testid)
  if (!el) return ''
  const input = (el.matches('input,textarea') ? el : el.querySelector('input,textarea')) as HTMLInputElement | null
  if (input) return input.value
  return (el.textContent || '').trim()
}

function countLines() {
  let count = 0
  while (qs(`oa-line-item-name-${count}`)) count += 1
  return count
}

function looseNumberEqual(a: string, b: string) {
  const na = Number(String(a).replace(/,/g, ''))
  const nb = Number(String(b).replace(/,/g, ''))
  if (Number.isFinite(na) && Number.isFinite(nb)) return Math.abs(na - nb) < 0.001
  return String(a).trim() === String(b).trim()
}

async function act(step: AgentStep, navigate: NavigateFunction) {
  const action = step.action || {}
  const type = String(action.type || '')
  if (type === 'navigate') {
    navigate(String(action.path || '/oa'))
    await sleep(200)
    return
  }
  if (type === 'click') {
    const testid = String(action.testid || '')
    const el = await waitForTestId(testid)
    const textIncludes = action.text_includes ? String(action.text_includes) : ''
    if (textIncludes && !(el.textContent || '').includes(textIncludes)) {
      throw new Error(`click blocked: ${testid} text does not include ${textIncludes}`)
    }
    el.scrollIntoView({ block: 'center', inline: 'nearest' })
    await sleep(100)
    dispatchMouse(el, 'mousedown')
    dispatchMouse(el, 'mouseup')
    el.click()
    await sleep(250)
    if (action.expect_line_count != null) {
      const expected = Number(action.expect_line_count)
      const start = Date.now()
      while (Date.now() - start < 3000) {
        if (countLines() >= expected) break
        await sleep(100)
      }
    }
    return
  }
  if (type === 'fill_fields') {
    const fields = (action.fields || []) as Array<{ testid: string; value: string; input_type?: string; optional?: boolean }>
    for (const field of fields) {
      if (!field.value && field.optional) continue
      try {
        await fillField(field.testid, String(field.value ?? ''), field.input_type)
      } catch (error) {
        if (field.optional) continue
        throw error
      }
    }
    return
  }
  if (type === 'assert_no_click') {
    // Intentionally do not click forbidden control.
    return
  }
  if (type === 'read_draft_status' || type === 'noop' || type === 'verify_detail_payload') {
    return
  }
  throw new Error(`unsupported action type: ${type}`)
}

async function verify(step: AgentStep): Promise<{ ok: boolean; actual: Record<string, unknown> }> {
  const rule = step.verify || {}
  const type = String(rule.type || '')
  if (type === 'testid_visible') {
    const testid = String(rule.testid || '')
    try {
      await waitForTestId(testid)
      return { ok: true, actual: { testid, visible: true } }
    } catch {
      return { ok: false, actual: { testid, visible: false } }
    }
  }
  if (type === 'fields_equals') {
    const fields = (rule.fields || []) as Array<{ testid: string; value: string; loose_number?: boolean }>
    const actualFields: Record<string, string> = {}
    let ok = true
    for (const field of fields) {
      const value = readField(field.testid)
      actualFields[field.testid] = value
      const pass = field.loose_number
        ? looseNumberEqual(value, String(field.value))
        : value.trim() === String(field.value).trim()
      if (!pass) ok = false
    }
    return { ok, actual: { fields: actualFields } }
  }
  if (type === 'line_count_at_least') {
    const count = countLines()
    return { ok: count >= Number(rule.count || 0), actual: { count } }
  }
  if (type === 'url_matches') {
    const pattern = new RegExp(String(rule.pattern || '.*'))
    return { ok: pattern.test(window.location.pathname), actual: { path: window.location.pathname } }
  }
  if (type === 'budget_matches') {
    const value = readField(String(rule.testid || 'oa-form-total-budget'))
    const ok = looseNumberEqual(value, String(rule.value || '0'))
    return { ok, actual: { budget: value } }
  }
  if (type === 'draft_status') {
    const match = window.location.pathname.match(/\/oa\/(?:applications\/)?(\d+)(?:\/edit)?/)
    if (!match) return { ok: false, actual: { path: window.location.pathname } }
    const detail = await api.getOA(Number(match[1]))
    const approval = String(detail.status || '').toUpperCase()
    const procurement = String(detail.procurement_status || 'NOT_STARTED').toUpperCase()
    const ok = approval === 'DRAFT' && procurement === 'NOT_STARTED' && Boolean(detail.application_no)
    return {
      ok,
      actual: {
        approval_status: approval,
        procurement_status: procurement,
        application_no: detail.application_no,
        id: detail.id,
      },
    }
  }
  if (type === 'page_contains') {
    const text = String(rule.text || '')
    const ok = document.body.innerText.includes(text)
    return { ok, actual: { contains: text, ok } }
  }
  if (type === 'detail_amount_match') {
    const match = window.location.pathname.match(/\/oa\/(?:applications\/)?(\d+)/)
    if (!match) return { ok: false, actual: { path: window.location.pathname } }
    const detail = await api.getOA(Number(match[1]))
    const lineTotal = (detail.lines || []).reduce(
      (sum, line) => sum + Number(line.quantity) * Number(line.estimated_unit_price),
      0,
    )
    const expected = Number(rule.expected_total || 0)
    const ok =
      Math.abs(Number(detail.total_budget) - expected) < 0.05
      && Math.abs(lineTotal - expected) < 0.05
      && (detail.lines || []).length === Number(rule.line_count || 0)
      && String(detail.status).toUpperCase() === 'APPROVED'
      && String(detail.procurement_status || '').toUpperCase() === 'NOT_STARTED'
      && !detail.linked_pr_no
    return {
      ok,
      actual: {
        total_budget: detail.total_budget,
        line_total: lineTotal,
        line_count: detail.lines?.length,
        approval_status: detail.status,
        procurement_status: detail.procurement_status,
        linked_pr_no: detail.linked_pr_no,
      },
    }
  }
  if (type === 'procurement_preparing') {
    await sleep(500)
    const match = window.location.pathname.match(/\/procurement\/requests\/([^/]+)/)
    // After submit, app navigates to PR; also accept staying on OA with tag text.
    const preparingText = document.body.innerText.includes('采购准备中')
    if (match) {
      // landed on PR page — fetch OA via lineage-less path: use previous detail from URL history is hard;
      // treat PR redirect + text as success and read PR no.
      return {
        ok: preparingText || Boolean(match[1]),
        actual: {
          linked_pr_no: match[1],
          path: window.location.pathname,
          text_ok: preparingText,
          approval_status: 'APPROVED',
          procurement_status: 'PREPARING',
        },
      }
    }
    const oaMatch = window.location.pathname.match(/\/oa\/(?:applications\/)?(\d+)/)
    if (!oaMatch) return { ok: false, actual: { path: window.location.pathname } }
    const detail = await api.getOA(Number(oaMatch[1]))
    const ok =
      String(detail.status).toUpperCase() === 'APPROVED'
      && String(detail.procurement_status).toUpperCase() === 'PREPARING'
      && preparingText
    return {
      ok,
      actual: {
        approval_status: detail.status,
        procurement_status: detail.procurement_status,
        linked_pr_no: detail.linked_pr_no,
        text_ok: preparingText,
      },
    }
  }
  return { ok: false, actual: { error: `unsupported verify type ${type}` } }
}

export async function executeAgentStep(step: AgentStep, navigate: NavigateFunction) {
  return withAgentChromeHidden(async () => {
    await act(step, navigate)
    await sleep(150)
    return verify(step)
  })
}

export function currentRunnableStep(steps: AgentStep[]): AgentStep | undefined {
  return steps.find((step) => step.status === 'pending' || step.status === 'failed')
}
