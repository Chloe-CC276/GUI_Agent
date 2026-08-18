import type { OAStatus, ProcurementStatus } from './types'

export const API_BASE = '/api/v1'

export const oaStatusOptions: Array<{ value: OAStatus; label: string; color: string }> = [
  { value: 'DRAFT', label: '草稿', color: 'default' },
  { value: 'PENDING_APPROVAL', label: '待审批', color: 'blue' },
  { value: 'IN_APPROVAL', label: '审批中', color: 'processing' },
  { value: 'APPROVED', label: '已通过', color: 'success' },
  { value: 'REJECTED', label: '已驳回', color: 'error' },
]

export const procurementStatusOptions: Array<{ value: ProcurementStatus; label: string; color: string }> = [
  { value: 'NOT_STARTED', label: '未开始', color: 'default' },
  { value: 'PREPARING', label: '采购准备中', color: 'cyan' },
  { value: 'AWARDED', label: '已定标', color: 'purple' },
]

/** 旧小写 / 过渡状态 → 规范审批状态 */
export const oaLegacyStatusMap: Record<string, OAStatus> = {
  draft: 'DRAFT',
  pending_approval: 'PENDING_APPROVAL',
  pending: 'IN_APPROVAL',
  approving: 'IN_APPROVAL',
  approved: 'APPROVED',
  rejected: 'REJECTED',
  // Legacy mixed labels that used to overwrite approval status.
  procurement_prep: 'APPROVED',
  PROCUREMENT_PREP: 'APPROVED',
  awarded: 'APPROVED',
  AWARDED: 'APPROVED',
  order_created: 'APPROVED',
  ORDER_CREATED: 'APPROVED',
}

export const procurementLegacyStatusMap: Record<string, ProcurementStatus> = {
  not_started: 'NOT_STARTED',
  preparing: 'PREPARING',
  procurement_prep: 'PREPARING',
  PROCUREMENT_PREP: 'PREPARING',
  awarded: 'AWARDED',
  AWARDED: 'AWARDED',
  order_created: 'AWARDED',
  ORDER_CREATED: 'AWARDED',
}

export const awardSourceOptions = [
  { value: 'offline_inquiry', label: '线下询比价' },
  { value: 'framework', label: '框架协议' },
  { value: 'mall', label: '商城' },
  { value: 'direct', label: '直接采购' },
  { value: 'other', label: '其他' },
]

export const oaUrgencyOptions = [
  { value: 'NORMAL', label: '普通' },
  { value: 'URGENT', label: '加急' },
  { value: 'CRITICAL', label: '特急' },
]

export const oaRequestedMethodOptions = [
  { value: 'online', label: '网购' },
  { value: 'inquiry', label: '比价' },
  { value: 'bidding', label: '招标' },
  { value: 'centralized', label: '集中采购' },
  { value: 'single', label: '单一来源' },
  { value: 'framework', label: '框架协议' },
]

/** @deprecated 兼容旧导航引用；主导航已改为三大系统标签 */
export const navigation = [
  { key: '/oa', label: 'OA 申请' },
  { key: '/procurement', label: '采购云' },
  { key: '/erp/materials', label: 'ERP 物料' },
  { key: '/erp/orders', label: 'ERP 订单' },
  { key: '/agent', label: 'GUI Agent' },
]

export type SystemId = 'oa' | 'procurement' | 'erp'

export interface SystemSubPage {
  key: string
  label: string
  path: string
  testId: string
  legacyNavId?: string
}

export interface SystemTab {
  id: SystemId
  label: string
  accent: string
  defaultPath: string
  match: (pathname: string) => boolean
  children: SystemSubPage[]
}

export const systemTabs: SystemTab[] = [
  {
    id: 'oa',
    label: 'OA 系统',
    accent: 'oa',
    defaultPath: '/oa',
    match: (pathname) => pathname === '/oa' || pathname.startsWith('/oa/'),
    children: [
      { key: 'list', label: '申请列表', path: '/oa', testId: 'subnav-oa-list', legacyNavId: 'nav-oa' },
      { key: 'approvals', label: '审批工作台', path: '/oa/approvals', testId: 'subnav-oa-approvals' },
    ],
  },
  {
    id: 'procurement',
    label: '采购云',
    accent: 'procurement',
    defaultPath: '/procurement',
    match: (pathname) => pathname === '/procurement' || pathname.startsWith('/procurement/'),
    children: [
      { key: 'list', label: '采购申请', path: '/procurement', testId: 'subnav-pr-list', legacyNavId: 'nav-procurement' },
    ],
  },
  {
    id: 'erp',
    label: 'ERP',
    accent: 'erp',
    defaultPath: '/erp/workbench',
    match: (pathname) => pathname === '/erp' || pathname.startsWith('/erp/'),
    children: [
      { key: 'workbench', label: '采购工作台', path: '/erp/workbench', testId: 'subnav-erp-workbench', legacyNavId: 'nav-erp-workbench' },
      { key: 'po-candidates', label: '待建 PO', path: '/erp/po-candidates', testId: 'subnav-erp-po-candidates' },
      { key: 'orders', label: '采购订单', path: '/erp/orders', testId: 'subnav-erp-orders', legacyNavId: 'nav-erp-orders' },
      { key: 'dashboard', label: '看板', path: '/erp/dashboard', testId: 'subnav-erp-dashboard' },
      { key: 'materials', label: '物料主数据', path: '/erp/materials', testId: 'subnav-erp-materials', legacyNavId: 'nav-erp-materials' },
      { key: 'requests-new', label: '新建采购申请', path: '/erp/requests/new', testId: 'subnav-erp-request-new' },
      { key: 'export', label: '批量导出与核对', path: '/erp/export', testId: 'subnav-erp-export' },
    ],
  },
]

export const WORKBENCH_STAGES = [
  '需求填报',
  '部门审批',
  '预算校验',
  '采购寻源',
  '供应商报价',
  '定标下单',
  '合同签署',
  '到货验收',
  '财务付款',
] as const

/** Align with OA requested_method so PR can default from OA and remain editable. */
export const purchaseTypeOptions = [
  { value: 'online', label: '网购' },
  { value: 'inquiry', label: '询比价' },
  { value: 'bidding', label: '招标' },
  { value: 'centralized', label: '集中采购' },
  { value: 'single', label: '单一来源' },
  { value: 'framework', label: '框架协议' },
]

export const procurementHeaderFields = [
  { name: 'title', label: '采购单标题', required: true, placeholder: '请输入采购单标题' },
  { name: 'department', label: '采购部门', required: true, placeholder: '请输入采购部门' },
  { name: 'buyer', label: '采购负责人', required: true, placeholder: '请输入负责人' },
] as const

export const erpRequestHeaderFields = [
  { name: 'department', label: '申请部门', required: true },
  { name: 'applicant', label: '申请人', required: true },
  { name: 'budget_project', label: '预算项目', required: true },
  { name: 'cost_center', label: '成本中心', required: true },
  { name: 'purchase_type', label: '采购类型', required: true },
  { name: 'expected_delivery_date', label: '期望到货日', required: true },
  { name: 'receive_address', label: '收货地址', required: true },
  { name: 'purchase_reason', label: '采购原因', required: true },
] as const

export const resolveSystem = (pathname: string): SystemId => {
  const hit = systemTabs.find((tab) => tab.match(pathname))
  return hit?.id ?? 'oa'
}

export const resolveActiveSubPath = (pathname: string, system: SystemId): string => {
  const tab = systemTabs.find((item) => item.id === system)
  if (!tab) return '/oa'
  if (system === 'oa') {
    if (pathname === '/oa/approvals' || pathname.startsWith('/oa/approvals/')) return '/oa/approvals'
    return '/oa'
  }
  if (system === 'procurement') return '/procurement'
  if (system === 'erp') {
    if (pathname.startsWith('/erp/orders/') || pathname.startsWith('/erp/pos/')) return '/erp/orders'
    if (pathname.startsWith('/erp/po-create/')) return '/erp/po-candidates'
    if (pathname.startsWith('/erp/po-candidates')) return '/erp/po-candidates'
    if (pathname.startsWith('/erp/dashboard')) return '/erp/dashboard'
  }
  const exact = tab.children.find((child) => child.path === pathname)
  if (exact) return exact.path
  return tab.defaultPath
}
