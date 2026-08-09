import {
  oaLegacyStatusMap,
  oaStatusOptions,
  procurementLegacyStatusMap,
  procurementStatusOptions,
} from '../config'
import type {
  OARequest,
  OAStatus,
  OAStatusInput,
  ProcurementLine,
  ProcurementStatus,
  ProcurementStatusInput,
} from '../types'

export const normalizeOAStatus = (status?: OAStatusInput | null): OAStatus | undefined => {
  if (status == null || status === '') return undefined
  const raw = String(status)
  if (oaLegacyStatusMap[raw]) return oaLegacyStatusMap[raw]
  const upper = raw.toUpperCase() as OAStatus
  if (oaStatusOptions.some((item) => item.value === upper)) return upper
  const lower = raw.toLowerCase()
  return oaLegacyStatusMap[lower]
}

export const normalizeProcurementStatus = (
  status?: ProcurementStatusInput | null,
): ProcurementStatus | undefined => {
  if (status == null || status === '') return undefined
  const raw = String(status)
  if (procurementLegacyStatusMap[raw]) return procurementLegacyStatusMap[raw]
  const upper = raw.toUpperCase() as ProcurementStatus
  if (procurementStatusOptions.some((item) => item.value === upper)) return upper
  return procurementLegacyStatusMap[raw.toLowerCase()]
}

/** Resolve display status: submitted draft → 待审批, never show 草稿 after submit. */
export const resolveOADisplayStatus = (
  status?: OAStatusInput | null,
  isSubmitted?: boolean | null,
): OAStatus | undefined => {
  const normalized = normalizeOAStatus(status)
  if (normalized === 'DRAFT' && isSubmitted) return 'PENDING_APPROVAL'
  return normalized
}

export const getOAStatusMeta = (
  status?: OAStatusInput | null,
  isSubmitted?: boolean | null,
) => {
  const resolved = resolveOADisplayStatus(status, isSubmitted)
  if (!resolved) {
    return { value: (status || '') as OAStatus, label: String(status || '-'), color: 'default' }
  }
  return oaStatusOptions.find((item) => item.value === resolved) ?? {
    value: resolved,
    label: resolved,
    color: 'default',
  }
}

export const getProcurementStatusMeta = (status?: ProcurementStatusInput | null) => {
  const resolved = normalizeProcurementStatus(status) || 'NOT_STARTED'
  return procurementStatusOptions.find((item) => item.value === resolved) ?? {
    value: resolved,
    label: resolved,
    color: 'default',
  }
}

export const canEnterProcurement = (
  status?: OAStatusInput | null,
  procurementStatus?: ProcurementStatusInput | null,
  linkedPoNo?: string | null,
) => {
  if (linkedPoNo) return false
  if (normalizeOAStatus(status) !== 'APPROVED') return false
  const proc = normalizeProcurementStatus(procurementStatus) || 'NOT_STARTED'
  return proc === 'NOT_STARTED' || proc === 'PREPARING'
}

export const canEditOA = (status?: OAStatusInput | null, isSubmitted?: boolean | null) => {
  const resolved = resolveOADisplayStatus(status, isSubmitted)
  return resolved === 'DRAFT' || resolved === 'REJECTED'
}

export const isPendingApproval = (record: Pick<OARequest, 'status' | 'is_submitted'>) =>
  resolveOADisplayStatus(record.status, record.is_submitted) === 'PENDING_APPROVAL'

export const isOAStateConflict = (error: unknown) => {
  const status = (error as { response?: { status?: number } })?.response?.status
  const message = error instanceof Error ? error.message : String(error || '')
  return status === 409 || /STATE_CONFLICT|状态已变化|row_version|乐观锁/i.test(message)
}

export const lineAmount = (line: ProcurementLine) =>
  Number(line.quantity || 0) * Number(line.unit_price || 0)

export const totalAmount = (lines: ProcurementLine[] = []) =>
  lines.reduce((sum, line) => sum + lineAmount(line), 0)
