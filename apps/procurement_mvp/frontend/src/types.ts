export type OAStatus =
  | 'DRAFT'
  | 'PENDING_APPROVAL'
  | 'IN_APPROVAL'
  | 'APPROVED'
  | 'REJECTED'
export type ProcurementStatus = 'NOT_STARTED' | 'PREPARING' | 'AWARDED'
/** 兼容后端过渡期返回的旧小写/混写状态 */
export type OAStatusInput =
  | OAStatus
  | 'draft'
  | 'pending'
  | 'approving'
  | 'approved'
  | 'rejected'
  | 'PROCUREMENT_PREP'
  | 'AWARDED'
  | 'ORDER_CREATED'
  | string
export type ProcurementStatusInput = ProcurementStatus | 'procurement_prep' | 'order_created' | string
export type DecimalValue = number | string
export type OAUrgencyLevel = 'NORMAL' | 'URGENT' | 'CRITICAL'
export type ApprovalAction = 'SUBMIT' | 'START' | 'APPROVE' | 'REJECT' | 'RESUBMIT'
export type ApprovalQueue = 'pending_start' | 'in_approval' | 'done'

export interface Material {
  material_code: string
  material_name: string
  specification: string
  unit: string
  standard_price: DecimalValue
  status: string
}

export interface OALine {
  id?: number
  item_name: string
  specification?: string
  quantity: DecimalValue
  estimated_unit_price: DecimalValue
}

export interface ApprovalHistory {
  id?: number
  oa_apply_no?: string
  oa_version?: number
  action: ApprovalAction | string
  from_status?: string
  to_status?: string
  operator_id?: string
  operator_name?: string
  opinion?: string
  snapshot_json?: Record<string, unknown> | string
  created_at: string
}

export interface OARequest {
  id: number
  application_no: string
  title: string
  applicant: string
  department: string
  status: OAStatus | OAStatusInput
  /** PRD alias，与 status 同义 */
  approval_status?: OAStatus | OAStatusInput
  total_budget: DecimalValue
  created_at: string
  updated_at?: string
  lines: OALine[]
  attachments?: Array<AttachmentPayload & { id?: number; source_attachment_id?: string; size?: number; mime_type?: string }>
  oa_version?: number
  row_version?: number
  is_submitted?: boolean
  submitted_at?: string
  approval_started_at?: string
  approved_time?: string
  current_approver_id?: string
  current_approver_name?: string
  approved_by?: string
  approval_opinion?: string
  purchase_reason?: string
  budget_project_code?: string
  budget_project_name?: string
  cost_center_code?: string
  requested_method?: string
  urgency_level?: OAUrgencyLevel | string
  expected_completion_date?: string
  remark?: string
  approval_history?: ApprovalHistory[]
  /** PRD aliases */
  oa_apply_no?: string
  proposal_title?: string
  proposal_amount?: DecimalValue
  procurement_transfer_status?: string
  linked_pr_no?: string
  linked_po_no?: string
  erp_status?: string
  procurement_status?: ProcurementStatus | ProcurementStatusInput
  procurement_updated_at?: string
}

export interface OAApplicationPayload {
  title?: string
  applicant?: string
  department?: string
  total_budget?: number
  purchase_reason?: string
  budget_project_code?: string
  budget_project_name?: string
  cost_center_code?: string
  requested_method?: string
  urgency_level?: string
  expected_completion_date?: string | null
  remark?: string
  lines?: Array<{
    id?: number
    item_name: string
    specification?: string
    quantity: number
    estimated_unit_price: number
  }>
  attachments?: AttachmentPayload[]
  row_version?: number
}

export interface OAApprovalActionPayload {
  row_version: number
  opinion?: string
  reason?: string
  operator_id?: string
  operator_name?: string
}

export interface ProcurementLine {
  line_no: number
  oa_line_id?: number
  material_code?: string
  material_name?: string
  specification?: string
  unit?: string
  quantity?: DecimalValue
  unit_price?: DecimalValue
  raw_material_name?: string
  raw_specification?: string
  raw_unit?: string
  raw_quantity?: DecimalValue
  raw_estimated_unit_price?: DecimalValue
  match_confidence?: DecimalValue
  tax_rate?: DecimalValue
  budget_status?: string
}

export interface AttachmentPayload {
  file_name: string
  file_url: string
}

export interface Pagination {
  page: number
  page_size: number
  total: number
  pages: number
}

export interface PageResult<T> {
  items: T[]
  pagination: Pagination
}

export interface Supplier {
  supplier_code: string
  supplier_name: string
  status: string
  award_sources?: string[]
}

export interface AwardSource {
  code: string
  name: string
  status: string
}

export interface ProcurementRequest {
  id: number
  request_no: string
  oa_application_id: number
  status: string
  total_amount: DecimalValue
  lines: Array<ProcurementLine & { id: number; line_amount: DecimalValue; final_unit_price_tax?: DecimalValue }>
  attachments: Array<AttachmentPayload & { id: number }>
  oa_apply_no?: string
  oa_title?: string
  oa_applicant?: string
  oa_department?: string
  oa_total_budget?: DecimalValue
  oa_version?: number
  oa_approval_status?: string
  oa_approved_time?: string
  procurement_status?: ProcurementStatus | string
  po_no?: string
  erp_status?: string
  erp_sync_status?: string
  submission_version?: number
  procurement_transfer_status?: string
  submitted_at?: string
  created_at?: string
  updated_at?: string
  purchase_method?: string
  purchase_method_suggested?: string
  purchase_method_confirmed?: string
  oa_requested_method?: string
  rule_version?: string
  export_status?: string
  budget_project?: string
  cost_center?: string
  purchase_type?: string
  expected_delivery_date?: string
  receive_address?: string
  purchase_reason?: string
  supplier_code?: string
  supplier_name?: string
  award_source?: string
  final_total_amount_tax?: DecimalValue
  award_confirmed_by?: string
  award_confirmed_at?: string
  department?: string
  applicant?: string
}

export interface Transfer {
  transfer_id: string
  source_system: string
  source_key: string
  target_system: string
  target_key?: string
  transfer_type: string
  status: string
  phase: string
  idempotency_key: string
  retry_count: number
  error_code?: string
  error_message?: string
  task_id: string
  created_at: string
  updated_at: string
}

export interface Lineage {
  oa_apply_no: string
  pr_no?: string
  po_no?: string
  task_id?: string
  task_ids: string[]
  latest_status?: string
  transfers: Transfer[]
}

export interface ERPOrderLine {
  id: number
  line_no: number
  material_code: string
  material_name: string
  specification: string
  unit: string
  quantity: DecimalValue
  unit_price: DecimalValue
  line_amount: DecimalValue
}

export interface ERPOrder {
  po_no: string
  pr_no: string
  oa_apply_no?: string
  submission_version: number
  task_id: string
  status: string
  total_amount: DecimalValue
  created_at: string
  updated_at: string
  lines: ERPOrderLine[]
}

export interface TaskStatus {
  task_id: string
  business_key: string
  operation: string
  status: string
  result: Record<string, unknown>
  created_at: string
  current_route?: string
  context_json?: Record<string, unknown>
  is_paused?: boolean
}

export type StageState = 'done' | 'current' | 'todo' | 'unavailable' | 'completed' | 'pending'

export interface WorkbenchStage {
  index: number
  name: string
  state?: StageState
  status?: string
  available?: boolean
  route?: string
}

export interface WorkbenchMetric {
  key: string
  label: string
  value: number | string
  unit?: string
}

export interface WorkbenchSummary {
  stages: WorkbenchStage[]
  metrics: {
    pending?: number
    approving?: number
    month_amount?: DecimalValue
    budget_rate?: DecimalValue
    [key: string]: number | string | DecimalValue | undefined
  }
  current_pr?: ProcurementRequest | null
  current_stage?: number
  recent_events?: WorkflowEvent[]
}

export interface WorkflowEvent {
  id?: number | string
  business_key?: string
  event_type: string
  status?: string
  operator?: string
  event_time: string
  detail?: string
  detail_json?: Record<string, unknown>
}

export interface PurchaseMethodRule {
  method: string
  label: string
  min_amount: number
  max_amount?: number | null
}

export interface PurchaseMethodRules {
  version: string
  rules: PurchaseMethodRule[]
}

export interface ExportCandidate {
  request_no: string
  pr_no?: string
  department?: string
  applicant?: string
  content_summary?: string
  total_amount: DecimalValue
  status: string
  purchase_method_suggested?: string
  purchase_method_confirmed?: string
  validation_status?: 'pass' | 'passed' | 'review' | 'blocked' | string
  exportable?: boolean
  block_reason?: string
  oa_apply_no?: string
}

export interface BatchValidateResult {
  items: Array<{
    request_no: string
    pr_no?: string
    validation_status: string
    messages?: string[]
    purchase_method_suggested?: string
    purchase_method_confirmed?: string
  }>
  rule_version?: string
  summary?: Record<string, number>
}

export interface BatchExportResult {
  export_task_id: string
  file_name?: string
  file_url?: string
  status: string
  rule_version?: string
}

export interface ImportPreviewRow {
  row_no: number
  raw_material_name?: string
  raw_specification?: string
  material_code?: string
  material_name?: string
  quantity?: DecimalValue
  unit_price?: DecimalValue
  status: 'ok' | 'warning' | 'error' | string
  message?: string
  candidates?: Material[]
  match_confidence?: number
}

export interface ImportBatch {
  import_batch_id: string
  filename: string
  sheet_name?: string
  total_rows: number
  success_rows: number
  failed_rows: number
  status?: string
  rows?: ImportPreviewRow[]
}

export interface AgentActiveTask {
  task_id?: string
  business_key?: string
  operation?: string
  status?: string
  current_step?: string
  is_paused?: boolean
  context_json?: Record<string, unknown>
}
