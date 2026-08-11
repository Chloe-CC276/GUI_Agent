import axios, { AxiosError } from 'axios'
import { API_BASE } from './config'
import type {
  AgentActiveTask,
  ApprovalQueue,
  AttachmentPayload,
  AwardSource,
  BatchExportResult,
  BatchValidateResult,
  ERPOrder,
  ExportCandidate,
  ImportBatch,
  Lineage,
  Material,
  OAApplicationPayload,
  OAApprovalActionPayload,
  OARequest,
  PageResult,
  ProcurementLine,
  ProcurementRequest,
  PurchaseMethodRules,
  Supplier,
  TaskStatus,
  Transfer,
  WorkbenchSummary,
  WorkflowEvent,
} from './types'

export interface Envelope<T> {
  ok: boolean
  data: T
  task_id?: string
  business_key?: string
  idempotent_replay?: boolean
  error?: { code: string; message: string; details?: unknown }
}

export interface DraftPayload {
  task_id: string
  business_key: string
  oa_application_id?: number
  lines: Array<Omit<ProcurementLine, 'line_no'> & { material_code: string; line_amount?: number }>
  attachments: AttachmentPayload[]
  department?: string
  applicant?: string
  budget_project?: string
  cost_center?: string
  purchase_type?: string
  expected_delivery_date?: string
  receive_address?: string
  purchase_reason?: string
  title?: string
}

export const unwrapEnvelope = <T>(body: Envelope<T>): T => {
  if (!body.ok) throw new Error(body.error?.message || '请求失败')
  return body.data
}

export const extractApiError = (error: AxiosError<Envelope<unknown> | { detail?: string | Array<{ msg?: string }> }>) => {
  const body = error.response?.data
  if (body && 'error' in body && body.error?.message) return body.error.message
  const detail = body && 'detail' in body ? body.detail : undefined
  if (Array.isArray(detail)) return detail.map((item) => item.msg).filter(Boolean).join('；') || error.message
  return detail || error.message || '网络请求失败'
}

export const isApiUnavailable = (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)
  const status = (error as { response?: { status?: number } })?.response?.status
  if (status === 404 || status === 501 || status === 502 || status === 503) return true
  return /404|Not Found|Failed to fetch|Network Error|ECONNREFUSED|接口暂未|not implemented/i.test(message)
}

export const friendlyUnavailable = (feature: string, error?: unknown) => {
  const detail = error instanceof Error ? error.message : ''
  return `${feature}接口暂未就绪${detail ? `（${detail}）` : ''}，请稍后重试或确认后端已启动对应能力。`
}

export const client = axios.create({ baseURL: API_BASE, timeout: 15000 })
client.interceptors.response.use(
  (response) => {
    const body = response.data as Envelope<unknown>
    if (body && typeof body === 'object' && 'ok' in body) {
      response.data = unwrapEnvelope(body)
    }
    return response
  },
  (error: AxiosError<Envelope<unknown> | { detail?: string | Array<{ msg?: string }> }>) =>
    Promise.reject(Object.assign(new Error(extractApiError(error)), { response: error.response, isAxiosError: true })),
)

export const api = {
  listOA: (params?: {
    search?: string
    keyword?: string
    status?: string
    procurement_status?: string
    page?: number
    page_size?: number
  }) => {
    const { keyword, search, ...rest } = params || {}
    return client
      .get<PageResult<OARequest>>('/oa/applications', {
        params: { ...rest, search: search || keyword || undefined, keyword: keyword || search || undefined },
      })
      .then((r) => r.data)
  },
  getOA: (id: string | number) => client.get<OARequest>(`/oa/applications/${id}`).then((r) => r.data),
  createOA: (payload: OAApplicationPayload) =>
    client.post<OARequest>('/oa/applications', payload).then((r) => r.data),
  updateOA: (id: string | number, payload: OAApplicationPayload) =>
    client.put<OARequest>(`/oa/applications/${id}`, payload).then((r) => r.data),
  submitOA: (id: string | number, payload?: Partial<OAApprovalActionPayload>) =>
    client.post<OARequest>(`/oa/applications/${id}/submit`, payload || {}).then((r) => r.data),
  resubmitOA: (id: string | number, payload?: Partial<OAApprovalActionPayload>) =>
    client.post<OARequest>(`/oa/applications/${id}/resubmit`, payload || {}).then((r) => r.data),
  listOAApprovals: (params?: {
    queue?: ApprovalQueue | string
    search?: string
    keyword?: string
    status?: string
    page?: number
    page_size?: number
  }) => {
    const { keyword, search, ...rest } = params || {}
    return client
      .get<PageResult<OARequest>>('/oa/approvals', {
        params: { ...rest, search: search || keyword || undefined },
      })
      .then((r) => r.data)
  },
  startOAApproval: (id: string | number, payload: OAApprovalActionPayload) =>
    client.post<OARequest>(`/oa/applications/${id}/start-approval`, payload).then((r) => r.data),
  approveOA: (id: string | number, payload: OAApprovalActionPayload) =>
    client.post<OARequest>(`/oa/applications/${id}/approve`, payload).then((r) => r.data),
  rejectOA: (id: string | number, payload: OAApprovalActionPayload & { reason: string }) =>
    client.post<OARequest>(`/oa/applications/${id}/reject`, payload).then((r) => r.data),
  listApprovedOA: (params?: { since?: string; page?: number; page_size?: number }) =>
    client.get<PageResult<OARequest>>('/oa/applications/approved', { params }).then((r) => r.data),
  pushOA: (oaApplyNo: string, payload: { task_id: string; simulate_target_failure?: boolean; simulate_callback_failure?: boolean }) =>
    client.post<{ pr_no?: string; transfer: Transfer }>(`/oa/proposals/${oaApplyNo}/push-to-procurement`, payload).then((r) => r.data),
  submitProcurement: (oaApplyNo: string, payload: { task_id: string; simulate_target_failure?: boolean; simulate_callback_failure?: boolean }) =>
    client.post<{
      oa_apply_no: string
      approval_status: string
      procurement_status: string
      pr_no?: string
      redirect_url?: string
      transfer?: Transfer
    }>(`/oa/applications/${oaApplyNo}/submit-procurement`, payload).then((r) => r.data),
  getLineage: (oaApplyNo: string) =>
    client.get<Lineage>(`/oa/proposals/${oaApplyNo}/lineage`).then((r) => r.data),
  listMaterials: (params?: { search?: string; page?: number; page_size?: number }) =>
    client.get<PageResult<Material>>('/erp/materials', { params }).then((r) => r.data),
  listSuppliers: (params?: { search?: string }) =>
    client.get<{ items: Supplier[] }>('/erp/suppliers', { params }).then((r) => r.data),
  listAwardSources: () =>
    client.get<{ items: AwardSource[] }>('/erp/award-sources').then((r) => r.data),
  createProcurement: (payload: DraftPayload & { oa_application_id: number }) =>
    client.post<ProcurementRequest>('/procurement/requests', payload).then((r) => r.data),
  updateProcurementDraft: (id: number, payload: DraftPayload) =>
    client.put<ProcurementRequest>(`/procurement/requests/${id}/draft`, payload).then((r) => r.data),
  patchProcurement: (prNo: string, payload: Record<string, unknown>) =>
    client.patch<ProcurementRequest>(`/procurement/requests/${prNo}`, payload).then((r) => r.data),
  listProcurements: (params?: {
    pr_no?: string
    oa_apply_no?: string
    status?: string
    erp_status?: string
    procurement_status?: string
    q?: string
    search?: string
  }) =>
    client.get<PageResult<ProcurementRequest>>('/procurement/requests', { params }).then((r) => r.data),
  getProcurement: (reference: string | number) =>
    client.get<ProcurementRequest>(`/procurement/requests/${reference}`).then((r) => r.data),
  validateProcurement: (reference: string | number, payload: { task_id: string; business_key: string }) =>
    client.post<ProcurementRequest>(`/procurement/requests/${reference}/validate`, payload).then((r) => r.data),
  submitErp: (prNo: string, payload: { task_id: string; business_key: string; confirmed_by?: string; simulate_target_failure?: boolean; simulate_callback_failure?: boolean }) =>
    client.post<{
      po_no?: string
      pr_no?: string
      erp_sync_status?: string
      procurement_status?: string
      message?: string
      transfer?: Transfer
    }>(`/procurement/requests/${prNo}/submit-erp`, payload).then((r) => r.data),
  prepareERPSubmit: (prNo: string, payload: { task_id: string; business_key: string }) =>
    client.post<ProcurementRequest & { confirmation_token: string }>(`/procurement/requests/${prNo}/prepare-erp-submit`, payload).then((r) => r.data),
  pushToERP: (prNo: string, payload: { task_id: string; business_key: string; confirmation_token: string; simulate_target_failure?: boolean; simulate_callback_failure?: boolean }) =>
    client.post<{ po_no?: string; transfer: Transfer }>(`/procurement/requests/${prNo}/push-to-erp`, payload).then((r) => r.data),
  prepareSubmit: (id: number, payload: { task_id: string; business_key: string }) =>
    client.post<ProcurementRequest & { confirmation_token: string }>(`/procurement/requests/${id}/prepare-submit`, payload).then((r) => r.data),
  confirmSubmit: (id: number, payload: { task_id: string; business_key: string; confirmation_token: string }) =>
    client.post<ProcurementRequest>(`/procurement/requests/${id}/confirm-submit`, payload).then((r) => r.data),
  listERPOrders: (params?: { po_no?: string; pr_no?: string; oa_apply_no?: string; status?: string }) =>
    client.get<PageResult<ERPOrder>>('/erp/orders', { params }).then((r) => r.data),
  getERPOrder: (poNo: string) => client.get<ERPOrder>(`/erp/orders/${poNo}`).then((r) => r.data),
  listTransfers: (params?: { business_key?: string }) =>
    client.get<PageResult<Transfer>>('/integration/transfers', { params }).then((r) => r.data),
  retryTransfer: (transferId: string, payload?: { task_id?: string }) =>
    client.post<Transfer>(`/integration/transfers/${transferId}/retry`, payload).then((r) => r.data),
  getTask: (taskId: string) => client.get<TaskStatus>(`/agent/tasks/${taskId}`).then((r) => r.data),
  getAgentTask: (taskId: string) => client.get<TaskStatus>(`/agent/tasks/${taskId}`).then((r) => r.data),
  agentChat: (payload: {
    message: string
    route?: string
    business_key?: string
    folder_path?: string
    excel_path?: string
  }) =>
    client.post<{
      intent?: string
      reply?: string
      chips?: Array<{ id: string; label: string }>
      task?: TaskStatus | null
    }>('/agent/chat', payload).then((r) => r.data),
  continueAgentTask: (
    taskId: string,
    payload: { folder_path?: string; excel_path?: string; oa_id?: number; file?: File },
  ) => {
    const form = new FormData()
    if (payload.folder_path) form.append('folder_path', payload.folder_path)
    if (payload.excel_path) form.append('excel_path', payload.excel_path)
    if (payload.oa_id != null) form.append('oa_id', String(payload.oa_id))
    if (payload.file) form.append('file', payload.file)
    return client
      .post<TaskStatus>(`/agent/tasks/${taskId}/continue`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      .then((r) => r.data)
  },
  reportAgentStepResult: (
    taskId: string,
    payload: { step_id: string; status: 'passed' | 'failed'; actual?: unknown; detail?: Record<string, unknown> },
  ) =>
    client.post<TaskStatus>(`/agent/tasks/${taskId}/step-result`, payload).then((r) => r.data),
  resetDemo: () => client.post('/demo/reset').then((r) => r.data),

  getWorkbenchSummary: () =>
    client.get<WorkbenchSummary>('/workbench/summary').then((r) => r.data),
  getWorkbenchEvents: (params?: { business_key?: string }) =>
    client.get<WorkflowEvent[] | PageResult<WorkflowEvent>>('/workbench/events', { params }).then((r) => {
      const data = r.data
      if (Array.isArray(data)) return data
      return data?.items ?? []
    }),

  previewImport: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return client.post<ImportBatch>('/procurement/imports/preview', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then((r) => r.data)
  },
  confirmImport: (payload: {
    task_id: string
    business_key: string
    import_batch_id: string
    pr_no?: string
    oa_application_id?: number
    row_nos?: number[]
  }) =>
    client.post<{
      pr_no: string
      import_batch_id: string
      written_rows: number
      request: ProcurementRequest
      idempotent_replay?: boolean
    }>('/procurement/imports/confirm', payload).then((r) => r.data),

  listExportCandidates: (params?: Record<string, string | number | boolean | undefined>) =>
    client.get<{ items: Array<Record<string, unknown>>; total?: number } | ExportCandidate[]>(
      '/procurement/requests/export-candidates',
      { params },
    ).then((r) => {
      const raw = Array.isArray(r.data) ? r.data : (r.data?.items ?? [])
      const items = raw.map((item) => {
        const row = item as Record<string, unknown>
        const validation = (row.validation || {}) as { result?: string; blockers?: string[] }
        const prNo = String(row.pr_no || row.request_no || '')
        return {
          request_no: prNo,
          pr_no: prNo,
          department: row.department as string | undefined,
          applicant: row.applicant as string | undefined,
          content_summary: (row.title || row.content_summary || prNo) as string,
          total_amount: row.total_amount as string | number,
          status: String(row.status || ''),
          purchase_method_suggested: row.purchase_method_suggested as string | undefined,
          purchase_method_confirmed: row.purchase_method_confirmed as string | undefined,
          validation_status: validation.result || (row.validation_status as string | undefined) || row.export_status as string | undefined,
          exportable: validation.result ? validation.result !== 'blocked' : Boolean(row.exportable ?? true),
          block_reason: validation.blockers?.join(', '),
          oa_apply_no: row.oa_apply_no as string | undefined,
        } satisfies ExportCandidate
      })
      return {
        items,
        pagination: { page: 1, page_size: items.length, total: items.length, pages: 1 },
      }
    }),
  batchValidate: (payload: { pr_nos: string[] }) =>
    client.post<{
      rule_version?: string
      summary?: Record<string, number>
      items: Array<{
        pr_no: string
        purchase_method_suggested?: string
        purchase_method_confirmed?: string
        validation: { result: string; blockers?: string[]; warnings?: string[] }
      }>
    }>('/procurement/requests/batch-validate', payload).then((r) => ({
      rule_version: r.data.rule_version,
      summary: r.data.summary,
      items: (r.data.items || []).map((item) => ({
        request_no: item.pr_no,
        pr_no: item.pr_no,
        validation_status: item.validation?.result,
        messages: [...(item.validation?.blockers || []), ...(item.validation?.warnings || [])],
        purchase_method_suggested: item.purchase_method_suggested,
        purchase_method_confirmed: item.purchase_method_confirmed,
      })),
    } satisfies BatchValidateResult)),
  batchExport: (payload: {
    pr_nos: string[]
    template_version?: string
    filters?: Record<string, unknown>
  }) =>
    client.post<{
      export_task_id: string
      file_url?: string
      filename?: string
      rule_version?: string
      template_version?: string
    }>('/procurement/requests/batch-export', payload).then((r) => ({
      export_task_id: r.data.export_task_id,
      file_url: r.data.file_url,
      file_name: r.data.filename,
      status: 'generated',
      rule_version: r.data.rule_version,
    } satisfies BatchExportResult)),
  getPurchaseMethodRules: () =>
    client.get<{ version: string; rules: Array<{ method: string; min_amount: number; max_amount?: number }> }>(
      '/config/purchase-method-rules',
    ).then((r) => ({
      version: r.data.version,
      rules: (r.data.rules || []).map((rule) => ({
        method: rule.method,
        label: rule.method,
        min_amount: rule.min_amount,
        max_amount: rule.max_amount ?? null,
      })),
    } satisfies PurchaseMethodRules)),
  patchPurchaseMethod: (prNo: string, payload: { task_id?: string; purchase_method_confirmed: string }) =>
    client.patch<ProcurementRequest | Record<string, unknown>>(`/procurement/requests/${prNo}/purchase-method`, payload).then((r) => r.data),

  getAgentActive: () =>
    client.get<{ items: AgentActiveTask[] } | AgentActiveTask | null>('/agent/tasks/active').then((r) => {
      const data = r.data
      if (!data) return null
      if (Array.isArray((data as { items?: AgentActiveTask[] }).items)) {
        return (data as { items: AgentActiveTask[] }).items[0] ?? null
      }
      return data as AgentActiveTask
    }),
  pauseAgentTask: (taskId: string) =>
    client.post<TaskStatus>(`/agent/tasks/${taskId}/pause`).then((r) => r.data),
  resumeAgentTask: (taskId: string) =>
    client.post<TaskStatus>(`/agent/tasks/${taskId}/resume`).then((r) => r.data),
  stopAgentTask: (taskId: string) =>
    client.post<TaskStatus>(`/agent/tasks/${taskId}/stop`).then((r) => r.data),

  listPOCandidates: (params?: { status?: string; q?: string; page?: number; page_size?: number }) =>
    client.get<PageResult<Record<string, unknown> & { pr_no: string; status: string }>>('/erp/po-candidates', { params }).then((r) => r.data),
  createPOBatch: (payload: { pr_nos: string[]; operator?: string }) =>
    client.post<{
      batch_id: string
      tasks: Array<{ task_id?: string | null; pr_no: string; status: string; po_no?: string | null; route?: string }>
    }>('/erp/po-tasks/batch', payload).then((r) => r.data),
  runPOTask: (taskId: string) =>
    client.post<Record<string, unknown>>(`/erp/po-tasks/${taskId}/run`).then((r) => r.data),
  getPOTask: (taskId: string) =>
    client.get<Record<string, unknown>>(`/erp/po-tasks/${taskId}`).then((r) => r.data),
  getPOCreateContext: (reference: string) =>
    client.get<{
      task_id?: string | null
      pr_no: string
      status?: string
      po_no?: string | null
      steps?: Array<{ step_id: string; title?: string; status?: string }>
      form?: {
        header?: Record<string, unknown>
        lines?: Array<Record<string, unknown>>
        oa_apply_no?: string
        award_confirmed_at?: string
        purchase_method?: string
      }
      source?: {
        oa_apply_no?: string
        award_confirmed_at?: string
        purchase_method?: string
      }
    }>(`/erp/po-create-context/${reference}`).then((r) => r.data),
  createPOFromForm: (
    taskId: string,
    payload: { header: Record<string, unknown>; lines: Array<Record<string, unknown>>; simulate_readback_fail?: boolean },
  ) =>
    client.post<{ task_id: string; pr_no: string; po_no: string; status: string; order?: ERPOrder }>(
      `/erp/po-tasks/${taskId}/create-po`,
      payload,
    ).then((r) => r.data),
  markPOCreated: (
    taskId: string,
    payload: { po_no?: string; success?: boolean; error_code?: string; message?: string },
  ) => client.post<Record<string, unknown>>(`/erp/po-tasks/${taskId}/mark-created`, payload).then((r) => r.data),
  getPODetail: (poNo: string) =>
    client.get<{
      po_no: string
      pr_no: string
      oa_apply_no?: string
      status: string
      supplier_code?: string
      supplier_name?: string
      request_dept?: string
      purchasing_org?: string
      purchasing_group?: string
      currency_code?: string
      total_amount?: number | string
      total_amount_tax?: number | string
      batch_id?: string
      task_id?: string
      created_at?: string
      lines: Array<Record<string, unknown> & { id: number; line_no: number }>
      agent_summary?: {
        status?: string
        retry_count?: number
        takeover_flag?: boolean
        duration_ms?: number | null
        error_code?: string | null
        executor_type?: string | null
      }
    }>(`/erp/pos/${poNo}`).then((r) => r.data),
  getPOLineage: (poNo: string) =>
    client.get<Lineage & { batch_id?: string }>(`/erp/pos/${poNo}/lineage`).then((r) => r.data),
  getAgentDashboardSummary: (params?: { date_from?: string; date_to?: string; department?: string; batch_id?: string }) =>
    client.get<Record<string, number>>('/erp/agent-dashboard/summary', { params }).then((r) => r.data),
  getAgentDashboardFunnel: (params?: { date_from?: string; date_to?: string; batch_id?: string }) =>
    client.get<Record<string, number>>('/erp/agent-dashboard/funnel', { params }).then((r) => r.data),
  getAgentDashboardEvents: (params?: { event_type?: string; severity?: string; stage?: string; task_id?: string; page?: number; page_size?: number }) =>
    client.get<PageResult<Record<string, unknown>>>('/erp/agent-dashboard/events', { params }).then((r) => r.data),
  getAgentDashboardTasks: (params?: { status?: string; batch_id?: string; pr_no?: string; po_no?: string }) =>
    client.get<{ items: Array<Record<string, unknown>> }>('/erp/agent-dashboard/tasks', { params }).then((r) => r.data),
  getAgentDashboardTaskSteps: (taskId: string) =>
    client.get<{ items: Array<Record<string, unknown>> }>(`/erp/agent-dashboard/tasks/${taskId}/steps`).then((r) => r.data),
  retryAgentDashboardTask: (taskId: string) =>
    client.post<Record<string, unknown>>(`/erp/agent-dashboard/tasks/${taskId}/retry`).then((r) => r.data),
  stopAgentDashboardBatch: (batchId: string) =>
    client.post<Record<string, unknown>>(`/erp/agent-dashboard/batches/${batchId}/stop`).then((r) => r.data),
  getPODashboardSummary: (params?: { date_from?: string; date_to?: string; department?: string; supplier?: string }) =>
    client.get<Record<string, number | string>>('/erp/po-dashboard/summary', { params }).then((r) => r.data),
  getPODashboardTrend: (params?: { grain?: string; date_from?: string; date_to?: string }) =>
    client.get<{ items: Array<Record<string, unknown>> }>('/erp/po-dashboard/trend', { params }).then((r) => r.data),
  getPODashboardByDepartment: (params?: { metric?: string }) =>
    client.get<{ items: Array<Record<string, unknown>> }>('/erp/po-dashboard/by-department', { params }).then((r) => r.data),
  getPODashboardBySupplier: (params?: { limit?: number; metric?: string }) =>
    client.get<{ items: Array<Record<string, unknown>> }>('/erp/po-dashboard/by-supplier', { params }).then((r) => r.data),
  getPODashboardByMaterial: (params?: { limit?: number; metric?: string }) =>
    client.get<{ items: Array<Record<string, unknown>> }>('/erp/po-dashboard/by-material', { params }).then((r) => r.data),
  getPODashboardRecent: (params?: { page?: number; page_size?: number; status?: string }) =>
    client.get<PageResult<Record<string, unknown>>>('/erp/po-dashboard/recent-pos', { params }).then((r) => r.data),
}
