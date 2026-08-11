import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  App as AntApp,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Empty,
  Form,
  Input,
  InputNumber,
  Layout,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd'
import type { UploadFile } from 'antd'
import {
  CloudUploadOutlined,
  DeleteOutlined,
  PlusOutlined,
  RobotOutlined,
} from '@ant-design/icons'
import {
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
  useSearchParams,
} from 'react-router-dom'
import { api } from './api'
import { AgentFloatingWindow } from './components/AgentFloatingWindow'
import {
  awardSourceOptions,
  procurementHeaderFields,
  procurementStatusOptions,
  purchaseTypeOptions,
  resolveActiveSubPath,
  resolveSystem,
  systemTabs,
} from './config'
import { BatchExportPage } from './pages/BatchExportPage'
import { ERPRequestNewPage } from './pages/ERPRequestNewPage'
import { ERPBoardPage } from './pages/erp/ERPBoardPage'
import { ERPPOCandidateListPage } from './pages/erp/ERPPOCandidateListPage'
import { ERPPOCreatePage } from './pages/erp/ERPPOCreatePage'
import { ERPPODetailPage } from './pages/erp/ERPPODetailPage'
import { OAApprovalDetail } from './pages/oa/OAApprovalDetail'
import { OAApprovalWorkbench } from './pages/oa/OAApprovalWorkbench'
import { OADetailPage } from './pages/oa/OADetailPage'
import { OAFormPage } from './pages/oa/OAFormPage'
import { OAListPage } from './pages/oa/OAListPage'
import { WorkbenchPage } from './pages/WorkbenchPage'
import type { AttachmentPayload, ERPOrder, Lineage, Material, OARequest, ProcurementLine, ProcurementRequest, Supplier, TaskStatus, Transfer } from './types'
import { canEnterProcurement, getOAStatusMeta, getProcurementStatusMeta, lineAmount, totalAmount } from './utils/business'

const { Header, Content } = Layout
let taskSequence = 0
const nextTaskId = (operation: string) => `web-${operation}-${Date.now()}-${++taskSequence}`

export function EnterpriseLayout() {
  const navigate = useNavigate()
  const location = useLocation()
  const system = resolveSystem(location.pathname)
  const activeSub = resolveActiveSubPath(location.pathname, system)
  const currentTab = systemTabs.find((tab) => tab.id === system) || systemTabs[0]
  const [subnavVisible, setSubnavVisible] = useState(true)
  const [subnavKey, setSubnavKey] = useState(system)
  const previousSystem = useRef(system)

  useEffect(() => {
    if (previousSystem.current === system) {
      setSubnavVisible(true)
      return
    }
    previousSystem.current = system
    setSubnavVisible(false)
    const timer = window.setTimeout(() => {
      setSubnavKey(system)
      setSubnavVisible(true)
    }, 40)
    return () => window.clearTimeout(timer)
  }, [system])

  return (
    <Layout className="app-shell" data-testid="enterprise-layout">
      <Header className="app-topbar">
        <div className="brand brand-inline" data-testid="brand">
          <div className="brand-mark">企</div>
          <div><strong>智慧采购助手</strong><small>PROCUREMENT AI</small></div>
        </div>
        <nav className="system-tabs" data-testid="main-navigation">
          {systemTabs.map((tab) => (
            <button
              type="button"
              key={tab.id}
              className={`system-tab system-tab-${tab.accent}${system === tab.id ? ' is-active' : ''}`}
              data-testid={`system-tab-${tab.id}`}
              onClick={() => navigate(tab.defaultPath)}
            >
              <span className="system-tab-label">{tab.label}</span>
            </button>
          ))}
        </nav>
        <Typography.Text type="secondary" className="topbar-meta">安全 · 合规 · 可追溯</Typography.Text>
      </Header>
      <div
        key={subnavKey}
        className={`system-subnav system-subnav-${currentTab.accent}${subnavVisible ? ' is-open' : ''}`}
        data-testid="system-subnav"
      >
        <div className="system-subnav-inner">
          {currentTab.children.map((child) => (
            <button
              type="button"
              key={child.key}
              className={`system-subnav-item${activeSub === child.path ? ' is-active' : ''}`}
              data-testid={child.legacyNavId || child.testId}
              onClick={() => navigate(child.path)}
            >
              {child.label}
            </button>
          ))}
        </div>
      </div>
      <Content className="app-content"><Outlet /></Content>
      <AgentFloatingWindow />
    </Layout>
  )
}

function PageTitle({ title, subtitle, extra }: { title: string; subtitle: string; extra?: React.ReactNode }) {
  return (
    <div className="page-title">
      <div><Typography.Title level={3}>{title}</Typography.Title><Typography.Text type="secondary">{subtitle}</Typography.Text></div>
      {extra}
    </div>
  )
}

function MaterialsPage() {
  const [keyword, setKeyword] = useState('')
  const [rows, setRows] = useState<Material[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [total, setTotal] = useState(0)
  const load = async (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true)
    try {
      const data = await api.listMaterials({
        search: keyword || undefined,
        page: nextPage,
        page_size: nextPageSize,
      })
      setRows(data.items)
      setTotal(data.pagination?.total ?? data.items.length)
      setPage(data.pagination?.page ?? nextPage)
      setPageSize(data.pagination?.page_size ?? nextPageSize)
    } catch (e) { message.error((e as Error).message) }
    finally { setLoading(false) }
  }
  useEffect(() => { void load(1, pageSize) }, [])
  return (
    <>
      <PageTitle title="ERP 物料主数据" subtitle={`按编码、名称或规格检索企业物料（共 ${total} 条）`} />
      <Card>
        <Space className="filter-bar">
          <Input
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onPressEnter={() => void load(1, pageSize)}
            placeholder="物料编码 / 名称 / 规格"
            allowClear
            data-testid="erp-material-search"
          />
          <Button type="primary" onClick={() => void load(1, pageSize)} data-testid="erp-material-search-button">查询</Button>
        </Space>
        <Table
          rowKey="material_code"
          loading={loading}
          dataSource={rows}
          data-testid="erp-material-table"
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: [20, 50, 100, 200],
            showTotal: (value) => `共 ${value} 条`,
            onChange: (nextPage, nextPageSize) => {
              void load(nextPage, nextPageSize)
            },
          }}
          columns={[
            { title: '物料编码', dataIndex: 'material_code' },
            { title: '物料名称', dataIndex: 'material_name' },
            { title: '规格型号', dataIndex: 'specification' },
            { title: '计量单位', dataIndex: 'unit' },
            { title: '标准单价', dataIndex: 'standard_price', render: (v) => `¥${Number(v).toFixed(2)}` },
            { title: '状态', dataIndex: 'status', render: (v) => <Tag color={v === 'active' ? 'success' : 'default'}>{v}</Tag> },
          ]}
        />
      </Card>
    </>
  )
}

function LineageCard({ lineage, onRetry }: { lineage?: Lineage; onRetry?: (transfer: Transfer) => void }) {
  return (
    <Card title="来源链与传输" className="section-card" data-testid="lineage-card">
      {!lineage ? <Empty description="暂无追溯数据" /> : <>
        <Descriptions column={2}>
          <Descriptions.Item label="业务链">OA {lineage.oa_apply_no} → PR {lineage.pr_no || '-'} → PO {lineage.po_no || '-'}</Descriptions.Item>
          <Descriptions.Item label="最新状态">{lineage.latest_status || '-'}</Descriptions.Item>
          <Descriptions.Item label="task_ids" span={2}>{lineage.task_ids.join(', ') || '-'}</Descriptions.Item>
        </Descriptions>
        {lineage.transfers.map((transfer) => (
          <Alert
            key={transfer.transfer_id}
            className="transfer-row"
            type={transfer.status === 'success' ? 'success' : transfer.status === 'failed' || transfer.status === 'callback_failed' ? 'error' : 'info'}
            message={`${transfer.source_key} → ${transfer.target_key || transfer.target_system} · ${transfer.status}`}
            description={transfer.error_message || `phase: ${transfer.phase}`}
            action={onRetry && transfer.status !== 'success' ? <Button onClick={() => onRetry(transfer)} data-testid={`retry-transfer-${transfer.transfer_id}`}>retry</Button> : undefined}
          />
        ))}
      </>}
    </Card>
  )
}

function ProcurementListPage() {
  const navigate = useNavigate()
  const [q, setQ] = useState('')
  const [procurementStatus, setProcurementStatus] = useState<string>()
  const [rows, setRows] = useState<ProcurementRequest[]>([])
  const [loading, setLoading] = useState(false)
  const load = async () => {
    setLoading(true)
    try {
      setRows((await api.listProcurements({
        q: q || undefined,
        procurement_status: procurementStatus,
      })).items)
    } catch (e) { message.error((e as Error).message) }
    finally { setLoading(false) }
  }
  useEffect(() => { void load() }, [procurementStatus])
  return <>
    <PageTitle title="采购申请" subtitle="仅承接 OA 已通过且已提交采购的申请" />
    <Card>
      <Space wrap className="filter-bar">
        <Input
          placeholder="搜索 OA号/PR号/标题/申请人"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onPressEnter={() => void load()}
          data-testid="procurement-search-input"
        />
        <Select
          allowClear
          placeholder="采购状态"
          value={procurementStatus}
          onChange={(value) => setProcurementStatus(value)}
          options={procurementStatusOptions.filter((item) => item.value !== 'NOT_STARTED')}
          style={{ minWidth: 160 }}
          data-testid="procurement-status-filter"
        />
        <Button type="primary" onClick={() => void load()}>查询</Button>
      </Space>
      <Table
        rowKey="id"
        loading={loading}
        dataSource={rows}
        locale={{ emptyText: <Empty description="暂无待采购申请" /> }}
        data-testid="pr-table"
        columns={[
          { title: 'PR号', dataIndex: 'request_no' },
          { title: 'OA号', dataIndex: 'oa_apply_no' },
          { title: '标题', dataIndex: 'oa_title' },
          { title: '部门', dataIndex: 'oa_department' },
          { title: '申请人', dataIndex: 'oa_applicant' },
          { title: 'OA审批状态', render: () => <Tag color="success">已通过</Tag> },
          {
            title: '采购状态',
            dataIndex: 'procurement_status',
            render: (value) => {
              const meta = getProcurementStatusMeta(value)
              return <Tag color={meta.color}>{meta.label}</Tag>
            },
          },
          { title: '总金额', dataIndex: 'total_amount', render: (value) => `¥${Number(value).toFixed(2)}` },
          { title: '更新时间', dataIndex: 'updated_at' },
          { title: 'ERP PO号', dataIndex: 'po_no', render: (value) => value || '-' },
          {
            title: '操作',
            render: (_, row) => {
              const awarded = row.procurement_status === 'AWARDED' || Boolean(row.po_no)
              return (
                <Space>
                  <Button
                    type="link"
                    onClick={() => navigate(`/procurement/requests/${row.request_no}`)}
                    data-testid={awarded ? `procurement-view-${row.request_no}` : `procurement-continue-${row.request_no}`}
                  >
                    {awarded ? '查看详情' : '继续准备'}
                  </Button>
                  {row.po_no ? (
                    <Button
                      type="link"
                      onClick={() => navigate(`/erp/pos/${row.po_no}`)}
                      data-testid={`procurement-open-erp-${row.request_no}`}
                    >
                      查看ERP
                    </Button>
                  ) : null}
                </Space>
              )
            },
          },
        ]}
      />
    </Card>
  </>
}

function ProcurementDetailPage() {
  const { prNo = '' } = useParams()
  const navigate = useNavigate()
  const [record, setRecord] = useState<ProcurementRequest>()
  const [lineage, setLineage] = useState<Lineage>()
  const [materials, setMaterials] = useState<Material[]>([])
  const [suppliers, setSuppliers] = useState<Supplier[]>([])
  const [awardSourceCatalog, setAwardSourceCatalog] = useState(awardSourceOptions)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [loading, setLoading] = useState(true)
  const load = async () => {
    const data = await api.getProcurement(prNo)
    const method = data.purchase_method_confirmed || data.purchase_method || data.purchase_type || data.oa_requested_method
    setRecord(method ? { ...data, purchase_method_confirmed: method, purchase_method: method } : data)
    if (data.oa_apply_no) {
      try { setLineage(await api.getLineage(data.oa_apply_no)) } catch { setLineage(undefined) }
    }
  }
  useEffect(() => {
    load().catch((e) => message.error(e.message)).finally(() => setLoading(false))
    api.listSuppliers().then((result) => setSuppliers(result.items || [])).catch(() => undefined)
    api.listAwardSources().then((result) => {
      const items = result.items || []
      if (items.length) {
        setAwardSourceCatalog(items.map((item) => ({ value: item.code, label: item.name })))
      }
    }).catch(() => undefined)
  }, [prNo])
  const readOnly = Boolean(record?.po_no)
    || record?.procurement_status === 'AWARDED'
    || record?.erp_sync_status === 'WAITING_PO'
    || record?.status === 'submitted'
  const searchMaterials = async (search: string) => {
    try { setMaterials((await api.listMaterials({ search, page_size: 100 })).items.filter((item) => item.status === 'active')) }
    catch (e) { message.error((e as Error).message) }
  }
  const chooseMaterial = (lineId: number, material: Material) => setRecord((current) => current && ({
    ...current,
    lines: current.lines.map((line) => line.id === lineId ? { ...line, material_code: material.material_code, material_name: material.material_name, specification: material.specification, unit: material.unit, unit_price: material.standard_price } : line),
  }))
  const updateLinePrice = (lineId: number, unitPrice: number | null) => setRecord((current) => current && ({
    ...current,
    lines: current.lines.map((line) => {
      if (line.id !== lineId) return line
      const price = Number(unitPrice || 0)
      return { ...line, unit_price: price, line_amount: Number(line.quantity) * price }
    }),
  }))
  const savePrep = async () => {
    if (!record) return
    try {
      const method = record.purchase_method_confirmed
        || record.purchase_method
        || record.purchase_type
        || record.oa_requested_method
      const saved = await api.patchProcurement(record.request_no, {
        budget_project: record.budget_project,
        cost_center: record.cost_center,
        purchase_type: method,
        purchase_method: method,
        expected_delivery_date: record.expected_delivery_date,
        receive_address: record.receive_address,
        purchase_reason: record.purchase_reason,
        supplier_code: record.supplier_code,
        award_source: record.award_source,
        award_confirmed_by: '采购员',
        lines: record.lines.map((line) => ({
          id: line.id,
          material_code: line.material_code,
          material_name: line.material_name,
          specification: line.specification,
          unit: line.unit,
          quantity: line.quantity,
          unit_price: line.unit_price,
          final_unit_price_tax: line.unit_price,
        })),
      })
      setRecord(saved)
      message.success('采购准备已保存')
    } catch (e) { message.error((e as Error).message) }
  }
  const validate = async () => {
    if (!record) return
    try {
      await savePrep()
      const saved = await api.validateProcurement(record.request_no, {
        task_id: nextTaskId('validate'),
        business_key: record.oa_apply_no || record.request_no,
      })
      setRecord(saved)
      message.success('校验通过')
    } catch (e) { message.error((e as Error).message) }
  }
  const submitErp = async () => {
    if (!record) return
    try {
      setSubmitting(true)
      await savePrep()
      const result = await api.submitErp(record.request_no, {
        task_id: nextTaskId('submit-erp'),
        business_key: record.request_no,
        confirmed_by: '采购员',
      })
      setConfirmOpen(false)
      await load()
      if (result.po_no) {
        message.success(`ERP采购订单已创建：${result.po_no}`)
        navigate(`/erp/pos/${result.po_no}`)
      } else if (result.erp_sync_status === 'WAITING_PO') {
        message.success('已定标，已进入 ERP 待建 PO 列表')
        navigate('/erp/po-candidates')
      } else {
        message.success(result.message || '提交完成')
      }
    } catch (e) { message.error((e as Error).message) }
    finally { setSubmitting(false) }
  }
  const retry = async (transfer: Transfer) => {
    try { await api.retryTransfer(transfer.transfer_id, { task_id: nextTaskId('retry-transfer') }); await load() }
    catch (e) { message.error((e as Error).message) }
  }
  if (loading) return <Spin />
  if (!record) return <Empty description="未找到采购申请" />
  const finalTotal = Number(record.final_total_amount_tax ?? record.total_amount)
  const selectedSupplier = suppliers.find((item) => item.supplier_code === record.supplier_code)
  const supplierAwardCodes = selectedSupplier?.award_sources || []
  const awardOptionsForSupplier = awardSourceCatalog.filter(
    (item) => !supplierAwardCodes.length || supplierAwardCodes.includes(item.value),
  )
  const suppliersForAward = suppliers.filter(
    (item) => !record.award_source || !(item.award_sources?.length) || item.award_sources.includes(record.award_source),
  )
  const purchaseMethodValue = record.purchase_method_confirmed || record.purchase_method || record.purchase_type || record.oa_requested_method
  return <>
    <PageTitle
      title={`采购申请 ${record.request_no}`}
      subtitle="采购准备与最小采购结果确认"
      extra={record.po_no && <Button type="primary" onClick={() => navigate(`/erp/pos/${record.po_no}`)} data-testid="procurement-open-erp-detail">查看 ERP 订单</Button>}
    />
    <Card title="A. 来源信息" data-testid="procurement-source-card">
      <Descriptions column={3}>
        <Descriptions.Item label="OA号">{record.oa_apply_no || '-'}</Descriptions.Item>
        <Descriptions.Item label="标题">{record.oa_title || '-'}</Descriptions.Item>
        <Descriptions.Item label="部门">{record.oa_department || '-'}</Descriptions.Item>
        <Descriptions.Item label="申请人">{record.oa_applicant || '-'}</Descriptions.Item>
        <Descriptions.Item label="预算项目">{record.budget_project || '-'}</Descriptions.Item>
        <Descriptions.Item label="OA审批金额">¥{Number(record.oa_total_budget || 0).toFixed(2)}</Descriptions.Item>
        <Descriptions.Item label="OA建议采买方式">
          {purchaseTypeOptions.find((item) => item.value === record.oa_requested_method)?.label || record.oa_requested_method || '-'}
        </Descriptions.Item>
        <Descriptions.Item label="采购原因" span={2}>{record.purchase_reason || '-'}</Descriptions.Item>
        <Descriptions.Item label="审批通过时间">{record.oa_approved_time || '-'}</Descriptions.Item>
      </Descriptions>
    </Card>
    <Card title="B. PR基本信息" className="section-card" data-testid="procurement-header-card">
      <Descriptions column={3}>
        <Descriptions.Item label="PR号">{record.request_no}</Descriptions.Item>
        <Descriptions.Item label="采购方式">
          {readOnly ? (purchaseTypeOptions.find((item) => item.value === purchaseMethodValue)?.label || purchaseMethodValue || '-') : (
            <Select
              style={{ minWidth: 180 }}
              allowClear={false}
              placeholder="默认取自 OA，可修改"
              value={purchaseMethodValue || undefined}
              options={purchaseTypeOptions}
              data-testid="procurement-method-select"
              onChange={(value) => setRecord({
                ...record,
                purchase_method_confirmed: value,
                purchase_method: value,
                purchase_type: value,
              })}
            />
          )}
        </Descriptions.Item>
        <Descriptions.Item label="期望到货日">
          {readOnly ? (record.expected_delivery_date || '-') : (
            <Input
              value={record.expected_delivery_date || ''}
              data-testid="procurement-delivery-input"
              onChange={(e) => setRecord({ ...record, expected_delivery_date: e.target.value })}
            />
          )}
        </Descriptions.Item>
        <Descriptions.Item label="收货地址" span={2}>
          {readOnly ? (record.receive_address || '-') : (
            <Input
              value={record.receive_address || ''}
              data-testid="procurement-address-input"
              onChange={(e) => setRecord({ ...record, receive_address: e.target.value })}
            />
          )}
        </Descriptions.Item>
        <Descriptions.Item label="币种">CNY</Descriptions.Item>
      </Descriptions>
    </Card>
    <Card title="C. 物资明细" className="section-card" data-testid="procurement-lines-card">
      <Table rowKey="id" pagination={false} dataSource={record.lines} columns={[
        { title: '原始物资', render: (_, line) => `${line.raw_material_name || '-'} / ${line.raw_specification || '-'}` },
        {
          title: 'ERP物料',
          render: (_, line) => readOnly ? `${line.material_code} / ${line.material_name}` : (
            <Select
              showSearch
              filterOption={false}
              onSearch={searchMaterials}
              value={line.material_code || undefined}
              placeholder="匹配物料"
              data-testid={`pr-material-select-${line.id}`}
              options={materials.map((m) => ({ value: m.material_code, label: `${m.material_code} · ${m.material_name}`, material: m }))}
              onSelect={(_, option) => chooseMaterial(line.id, option.material as Material)}
            />
          ),
        },
        { title: '数量', dataIndex: 'quantity' },
        { title: '单位', dataIndex: 'unit' },
        {
          title: '含税单价',
          render: (_, line) => readOnly ? line.unit_price : (
            <InputNumber
              min={0}
              value={Number(line.unit_price)}
              data-testid={`pr-unit-price-${line.id}`}
              onChange={(value) => updateLinePrice(line.id, value)}
            />
          ),
        },
        { title: '行金额', dataIndex: 'line_amount' },
      ]} />
    </Card>
    <Card title="D. 附件与差异" className="section-card">
      {record.attachments.length ? record.attachments.map((item) => (
        <Typography.Link key={item.id} href={item.file_url}>{item.file_name}</Typography.Link>
      )) : <Empty description="暂无附件" />}
    </Card>
    <Card title="E. 采购结果确认" className="section-card" data-testid="procurement-award-card">
      <Descriptions column={2}>
        <Descriptions.Item label="最终供应商">
          {readOnly ? `${record.supplier_code || '-'} ${record.supplier_name || ''}` : (
            <Select
              style={{ minWidth: 260 }}
              showSearch
              optionFilterProp="label"
              allowClear
              placeholder="请选择最终供应商"
              value={record.supplier_code || undefined}
              data-testid="procurement-supplier-select"
              options={suppliersForAward.map((item) => ({
                value: item.supplier_code,
                label: `${item.supplier_code} · ${item.supplier_name}`,
              }))}
              onChange={(value) => {
                const hit = suppliers.find((item) => item.supplier_code === value)
                const allowed = hit?.award_sources || []
                const nextSource = record.award_source && allowed.includes(record.award_source)
                  ? record.award_source
                  : (allowed[0] || undefined)
                setRecord({
                  ...record,
                  supplier_code: value,
                  supplier_name: hit?.supplier_name,
                  award_source: nextSource,
                })
              }}
            />
          )}
        </Descriptions.Item>
        <Descriptions.Item label="结果来源">
          {readOnly ? (
            awardSourceCatalog.find((item) => item.value === record.award_source)?.label || record.award_source || '-'
          ) : (
            <Select
              style={{ minWidth: 200 }}
              allowClear
              placeholder={record.supplier_code ? '请选择结果来源' : '请先选择供应商'}
              value={record.award_source || undefined}
              options={awardOptionsForSupplier}
              data-testid="procurement-award-source-select"
              onChange={(value) => setRecord({ ...record, award_source: value })}
            />
          )}
        </Descriptions.Item>
        <Descriptions.Item label="最终采购总额">¥{finalTotal.toFixed(2)}</Descriptions.Item>
        <Descriptions.Item label="确认人">{record.award_confirmed_by || '-'}</Descriptions.Item>
      </Descriptions>
    </Card>
    <Card title="F. 提交区" className="section-card" data-testid="procurement-actions-card">
      <Descriptions>
        <Descriptions.Item label="PR状态">{record.status}</Descriptions.Item>
        <Descriptions.Item label="采购状态">{getProcurementStatusMeta(record.procurement_status).label}</Descriptions.Item>
        <Descriptions.Item label="ERP同步">{record.erp_sync_status || record.erp_status || '-'}</Descriptions.Item>
      </Descriptions>
      <Space>
        {!readOnly && <Button onClick={() => void savePrep()} data-testid="procurement-save-button">保存</Button>}
        {!readOnly && <Button onClick={() => void validate()} data-testid="validate-procurement-button">校验</Button>}
        {!readOnly && (
          <Button danger type="primary" onClick={() => setConfirmOpen(true)} data-testid="submit-erp-button">
            确认定标并进入待建PO
          </Button>
        )}
        {record.erp_sync_status === 'WAITING_PO' && !record.po_no && (
          <Button type="primary" onClick={() => navigate('/erp/po-candidates')} data-testid="open-po-candidates">
            打开待建 PO 列表
          </Button>
        )}
        {record.po_no && <Button onClick={() => navigate(`/erp/pos/${record.po_no}`)}>跳转ERP详情</Button>}
      </Space>
    </Card>
    <Card title="G. ERP结果" className="section-card" data-testid="procurement-erp-result-card">
      <Descriptions column={3}>
        <Descriptions.Item label="ERP PO号">
          <span data-testid="procurement-erp-po-no">{record.po_no || '-'}</span>
        </Descriptions.Item>
        <Descriptions.Item label="ERP同步状态">{record.erp_sync_status || record.erp_status || '-'}</Descriptions.Item>
        <Descriptions.Item label="创建时间">{record.submitted_at || '-'}</Descriptions.Item>
      </Descriptions>
    </Card>
    <LineageCard lineage={lineage} onRetry={retry} />
    <Modal
      open={confirmOpen}
      title="确认定标并进入待建 PO？"
      onCancel={() => setConfirmOpen(false)}
      onOk={() => void submitErp()}
      confirmLoading={submitting}
      okText="确认定标"
      okButtonProps={{ danger: true, 'data-testid': 'submit-erp-confirm-ok' } as React.ComponentProps<typeof Button>}
    >
      <Alert
        type="warning"
        showIcon
        message="高风险操作"
        description={(
          <div data-testid="submit-erp-confirm-summary">
            <div>确认后不会通过业务 API 直接创建 ERP PO。</div>
            <div>单据将进入 ERP「待建 PO」列表，由 GUI Agent 在 ERP 页面录入创建。</div>
            <div>OA：{record.oa_apply_no || '-'}</div>
            <div>PR：{record.request_no}</div>
            <div>采购方式：{purchaseTypeOptions.find((item) => item.value === purchaseMethodValue)?.label || purchaseMethodValue || '-'}</div>
            <div>供应商：{record.supplier_code} {record.supplier_name}</div>
            <div>最终总额：¥{finalTotal.toFixed(2)}</div>
            <div>物资行数：{record.lines.length}</div>
          </div>
        )}
      />
    </Modal>
  </>
}

function ERPOrderListPage() {
  const navigate = useNavigate()
  const [filters, setFilters] = useState({ po_no: '', pr_no: '', oa_apply_no: '', status: '' })
  const [rows, setRows] = useState<ERPOrder[]>([])
  const load = async () => {
    try { setRows((await api.listERPOrders(Object.fromEntries(Object.entries(filters).filter(([, value]) => value)) as typeof filters)).items) }
    catch (e) { message.error((e as Error).message) }
  }
  useEffect(() => { void load() }, [])
  return <>
    <PageTitle title="ERP 采购订单" subtitle="查询已由采购云传输至 ERP 的采购订单" />
    <Card>
      <Space wrap className="filter-bar">
        <Input placeholder="PO号" value={filters.po_no} onChange={(e) => setFilters({ ...filters, po_no: e.target.value })} onPressEnter={load} data-testid="erp-order-search" />
        <Input placeholder="PR号" value={filters.pr_no} onChange={(e) => setFilters({ ...filters, pr_no: e.target.value })} />
        <Input placeholder="OA号" value={filters.oa_apply_no} onChange={(e) => setFilters({ ...filters, oa_apply_no: e.target.value })} />
        <Select allowClear placeholder="状态" value={filters.status || undefined} onChange={(value) => setFilters({ ...filters, status: value || '' })} options={['created'].map((value) => ({ value, label: value }))} />
        <Button type="primary" onClick={load}>查询</Button>
      </Space>
      <Table rowKey="po_no" dataSource={rows} data-testid="erp-order-table" columns={[
        { title: 'PO', dataIndex: 'po_no' }, { title: 'PR', dataIndex: 'pr_no' }, { title: 'OA', dataIndex: 'oa_apply_no' },
        { title: '金额', dataIndex: 'total_amount', render: (value) => `¥${Number(value).toFixed(2)}` }, { title: '状态', dataIndex: 'status' }, { title: '时间', dataIndex: 'created_at' },
        { title: '操作', render: (_, row) => <Button type="link" onClick={() => navigate(`/erp/pos/${row.po_no}`)} data-testid={`erp-order-view-${row.po_no}`}>查看</Button> },
      ]} />
    </Card>
  </>
}

function ERPOrderDetailPage() {
  const { poNo = '' } = useParams()
  const navigate = useNavigate()
  const [order, setOrder] = useState<ERPOrder>()
  const [lineage, setLineage] = useState<Lineage>()
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.getERPOrder(poNo).then(async (data) => {
      setOrder(data)
      if (data.oa_apply_no) try { setLineage(await api.getLineage(data.oa_apply_no)) } catch { setLineage(undefined) }
    }).catch((e) => message.error(e.message)).finally(() => setLoading(false))
  }, [poNo])
  if (loading) return <Spin />
  if (!order) return <Empty description="未找到 ERP 订单" />
  return <>
    <PageTitle title={`ERP采购订单 ${order.po_no}`} subtitle="采购订单完整内容与来源链" extra={<Button onClick={() => navigate(`/procurement/${order.pr_no}`)}>返回PR详情</Button>} />
    <Card title="订单头"><Descriptions column={3}>
      <Descriptions.Item label="PO">{order.po_no}</Descriptions.Item><Descriptions.Item label="PR">{order.pr_no}</Descriptions.Item><Descriptions.Item label="OA">{order.oa_apply_no || '-'}</Descriptions.Item>
      <Descriptions.Item label="金额">¥{Number(order.total_amount).toFixed(2)}</Descriptions.Item><Descriptions.Item label="状态">{order.status}</Descriptions.Item><Descriptions.Item label="task_id">{order.task_id}</Descriptions.Item>
      <Descriptions.Item label="创建时间">{order.created_at}</Descriptions.Item><Descriptions.Item label="更新时间">{order.updated_at}</Descriptions.Item>
    </Descriptions></Card>
    <Card title="完整行明细" className="section-card"><Table rowKey="id" pagination={false} dataSource={order.lines} columns={[
      { title: '行号', dataIndex: 'line_no' }, { title: '物料编码', dataIndex: 'material_code' }, { title: '物料名称', dataIndex: 'material_name' },
      { title: '规格', dataIndex: 'specification' }, { title: '单位', dataIndex: 'unit' }, { title: '数量', dataIndex: 'quantity' }, { title: '单价', dataIndex: 'unit_price' }, { title: '金额', dataIndex: 'line_amount' },
    ]} /></Card>
    <LineageCard lineage={lineage} />
  </>
}

function ProcurementPage() {
  const [searchParams] = useSearchParams()
  const oaId = searchParams.get('oa_id')
  const [form] = Form.useForm()
  const [oa, setOA] = useState<OARequest>()
  const [oaLoading, setOALoading] = useState(Boolean(oaId))
  const [materials, setMaterials] = useState<Material[]>([])
  const [lines, setLines] = useState<ProcurementLine[]>([{ line_no: 1, quantity: 1 }])
  const [files, setFiles] = useState<UploadFile[]>([])
  const [request, setRequest] = useState<ProcurementRequest>()
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<{ request: ProcurementRequest; taskId: string; businessKey: string }>()

  useEffect(() => {
    if (!oaId) return
    api.getOA(oaId).then((data) => {
      setOA(data)
      form.setFieldsValue({ title: data.title, department: data.department })
      if (data.lines.length) setLines(data.lines.map((line, index) => ({
        line_no: index + 1,
        oa_line_id: line.id,
        material_name: line.item_name,
        specification: line.specification,
        quantity: line.quantity,
        unit_price: line.estimated_unit_price,
      })))
    }).catch((e) => message.error(e.message)).finally(() => setOALoading(false))
  }, [oaId, form])

  const blocked = !oaId || !oa || !canEnterProcurement(oa.status)
  const amount = useMemo(() => totalAmount(lines), [lines])
  const updateLine = (lineNo: number, patch: Partial<ProcurementLine>) =>
    setLines((current) => current.map((line) => line.line_no === lineNo ? { ...line, ...patch } : line))
  const addLine = () => setLines((current) => [...current, { line_no: Math.max(0, ...current.map((x) => x.line_no)) + 1, quantity: 1 }])
  const removeLine = (lineNo: number) => setLines((current) => current.filter((line) => line.line_no !== lineNo))
  const searchMaterials = async (search: string) => {
    try { setMaterials((await api.listMaterials({ search, page_size: 100 })).items.filter((item) => item.status === 'active')) }
    catch (e) { message.error((e as Error).message) }
  }

  const attachments = (): AttachmentPayload[] => files.map((file) => ({
    file_name: file.name,
    file_url: `demo://${encodeURIComponent(file.name)}`,
  }))

  const buildPayload = async (operation: string) => {
    await form.validateFields()
    if (!oa || !canEnterProcurement(oa.status)) throw new Error('必须从已审批通过的 OA 申请创建采购单')
    if (!lines.length || lines.some((line) => !line.material_code || !line.quantity || line.unit_price == null)) {
      throw new Error('请为每行匹配有效 ERP 物料，并填写数量和单价')
    }
    return {
      task_id: nextTaskId(operation),
      business_key: oa.application_no,
      oa_application_id: oa.id,
      lines: lines.map((line) => {
        const { line_no: _, ...payloadLine } = line
        return {
          ...payloadLine,
          material_code: line.material_code!,
          line_amount: lineAmount(line),
        }
      }),
      attachments: attachments(),
    }
  }

  const persistDraft = async (operation: string) => {
    const payload = await buildPayload(operation)
    let saved: ProcurementRequest
    if (request) {
      const { oa_application_id: _, ...updatePayload } = payload
      saved = await api.updateProcurementDraft(request.id, updatePayload)
    } else {
      saved = await api.createProcurement(payload)
    }
    setRequest(saved)
    return saved
  }

  const saveDraft = async () => {
    try {
      const saved = await persistDraft(request ? 'update-draft' : 'create-draft')
      message.success(`草稿已保存：${saved.request_no}`)
    }
    catch (e) { message.error((e as Error).message) }
  }

  const submit = async () => {
    try {
      setSubmitting(true)
      const saved = await persistDraft(request ? 'sync-draft' : 'create-before-submit')
      const businessKey = oa!.application_no
      const prepareTaskId = nextTaskId('prepare-submit')
      const prepared = await api.prepareSubmit(saved.id, { task_id: prepareTaskId, business_key: businessKey })
      if (!prepared.confirmation_token) throw new Error('prepare-submit 未返回 confirmation_token')
      const confirmTaskId = nextTaskId('confirm-submit')
      const confirmed = await api.confirmSubmit(saved.id, {
        task_id: confirmTaskId,
        business_key: businessKey,
        confirmation_token: prepared.confirmation_token,
      })
      setRequest(confirmed)
      setResult({ request: confirmed, taskId: confirmTaskId, businessKey })
      message.success('采购任务已提交')
    } catch (e) { message.error((e as Error).message) }
    finally { setSubmitting(false) }
  }
  const confirm = () => Modal.confirm({
    title: '确认提交采购任务？',
    content: '提交后 GUI Agent 将操作采购系统，属于高风险操作。请确认单据和金额无误。',
    okText: '确认并提交',
    cancelText: '取消',
    okButtonProps: { danger: true, 'data-testid': 'submit-confirm-ok' } as React.ComponentProps<typeof Button>,
    onOk: submit,
  })

  if (oaLoading) return <Spin />
  return (
    <>
      <PageTitle title="新建采购单" subtitle={oaId ? `承接 OA 申请：${oa?.application_no || oaId}` : '请先从已审批通过的 OA 申请进入'} />
      {blocked && <Alert type="error" showIcon message="采购操作已阻断" description={oa ? `当前 OA 状态：${getOAStatusMeta(oa.status).label}` : '缺少有效 OA 申请，请从 OA 详情进入。'} data-testid="procurement-block-alert" />}
      <Form form={form} layout="vertical" disabled={blocked} data-testid="procurement-form">
        <Card title="单据头">
          <Row gutter={20}>
            {procurementHeaderFields.map((field) => <Col span={8} key={field.name}><Form.Item name={field.name} label={field.label} rules={[{ required: field.required, message: `请填写${field.label}` }]}><Input placeholder={field.placeholder} data-testid={`header-${field.name}`} /></Form.Item></Col>)}
          </Row>
        </Card>
        <Card title="物资明细" className="section-card" extra={<Button icon={<PlusOutlined />} onClick={addLine} data-testid="add-material-line">新增一行</Button>}>
          <div className="line-grid line-grid-header"><span>行号</span><span>ERP 物料</span><span>规格</span><span>单位</span><span>数量</span><span>单价（元）</span><span>金额（元）</span><span>操作</span></div>
          {lines.map((line) => (
            <div className="line-grid" key={line.line_no} data-testid={`material-line-${line.line_no}`}>
              <strong>{line.line_no}</strong>
              <Select
                showSearch filterOption={false} onSearch={searchMaterials}
                value={line.material_code}
                placeholder={line.material_name ? `待匹配：${line.material_name}` : '输入编码/名称检索'}
                data-testid={`material-select-${line.line_no}`}
                options={materials.map((m) => ({ value: m.material_code, label: `${m.material_code} · ${m.material_name}`, material: m }))}
                onSelect={(_, option) => {
                  const material = option.material as Material
                  updateLine(line.line_no, { material_code: material.material_code, material_name: material.material_name, specification: material.specification, unit: material.unit, unit_price: material.standard_price })
                }}
              />
              <Input value={line.specification} onChange={(e) => updateLine(line.line_no, { specification: e.target.value })} data-testid={`line-spec-${line.line_no}`} />
              <Input value={line.unit} onChange={(e) => updateLine(line.line_no, { unit: e.target.value })} data-testid={`line-unit-${line.line_no}`} />
              <InputNumber min={0.0001} value={line.quantity} onChange={(v) => updateLine(line.line_no, { quantity: v || undefined })} data-testid={`line-quantity-${line.line_no}`} />
              <InputNumber min={0} precision={2} value={line.unit_price} onChange={(v) => updateLine(line.line_no, { unit_price: v || undefined })} data-testid={`line-price-${line.line_no}`} />
              <strong>¥{lineAmount(line).toFixed(2)}</strong>
              <Popconfirm title="删除该物资行？" onConfirm={() => removeLine(line.line_no)}><Button danger type="text" icon={<DeleteOutlined />} disabled={lines.length === 1} data-testid={`delete-line-${line.line_no}`} /></Popconfirm>
            </div>
          ))}
          <Divider />
          <div className="amount-summary" data-testid="total-amount"><Statistic title="采购总金额" value={amount} precision={2} prefix="¥" /></div>
          <Alert className="price-notice" type="info" showIcon message="价格权威说明" description="页面单价可编辑并用于预估；保存后服务端将以 ERP standard_price 重新计算，输入差异会留痕审计。" data-testid="server-price-notice" />
        </Card>
        <Card title="附件" className="section-card"><Upload.Dragger fileList={files} onChange={({ fileList }) => setFiles(fileList)} beforeUpload={() => false} multiple data-testid="attachment-upload"><CloudUploadOutlined className="upload-icon" /><p>附件仅生成 demo:// 引用，不上传二进制</p></Upload.Dragger></Card>
        <Card title="提交前校验" className="section-card" data-testid="validation-area">
          <Alert showIcon type={lines.every((l) => l.material_code && l.quantity && l.unit_price != null) ? 'success' : 'warning'} message="校验项" description="单据头必填；每行需选择 ERP 物料并填写数量、单价；OA 来源须已审批通过。" />
        </Card>
        {result && <Alert className="section-card" type="success" showIcon message={`采购单 ${result.request.request_no} 已提交`} description={`状态：${result.request.status}；task_id：${result.taskId}；business_key：${result.businessKey}`} data-testid="submit-result" />}
        <div className="form-actions">
          <Button onClick={saveDraft} disabled={blocked} data-testid="save-draft-button">保存草稿</Button>
          <Button type="primary" danger loading={submitting} onClick={confirm} disabled={blocked} data-testid="submit-procurement-button">提交采购任务</Button>
        </div>
      </Form>
    </>
  )
}

function AgentPage() {
  const [taskId, setTaskId] = useState('')
  const [task, setTask] = useState<TaskStatus>()
  const query = async () => {
    if (!taskId.trim()) return message.warning('请输入 task_id')
    try { setTask(await api.getTask(taskId.trim())) }
    catch (e) { message.error((e as Error).message) }
  }
  const reset = async () => {
    try { await api.resetDemo(); setTask(undefined); setTaskId(''); message.success('演示数据已重置') }
    catch (e) { message.error((e as Error).message) }
  }
  return (
    <>
      <PageTitle title="GUI Agent 操作中心" subtitle="查询采购自动化任务，查看业务关联与执行状态" />
      <Alert type="info" showIcon icon={<RobotOutlined />} message="GUI Agent 操作入口" description="采购单确认提交后，后端创建自动化任务。Agent 将根据 task_id 执行采购系统操作，并以 business_key 关联业务单据。" data-testid="agent-guide" />
      <Card title="任务查询" className="section-card">
        <Space.Compact block>
          <Input value={taskId} onChange={(e) => setTaskId(e.target.value)} onPressEnter={query} placeholder="输入 task_id" data-testid="agent-task-id-input" />
          <Button type="primary" onClick={query} data-testid="agent-query-button">查询状态</Button>
        </Space.Compact>
        {task && <Descriptions bordered column={2} className="task-result" data-testid="agent-task-result">
          <Descriptions.Item label="task_id">{task.task_id}</Descriptions.Item>
          <Descriptions.Item label="business_key">{task.business_key}</Descriptions.Item>
          <Descriptions.Item label="操作">{task.operation}</Descriptions.Item>
          <Descriptions.Item label="状态"><Tag color="processing">{task.status}</Tag></Descriptions.Item>
          <Descriptions.Item label="结果" span={2}><Typography.Text code>{JSON.stringify(task.result)}</Typography.Text></Descriptions.Item>
        </Descriptions>}
      </Card>
      <Card title="演示环境" className="section-card">
        <Typography.Paragraph type="secondary">清除演示任务与单据，恢复预置数据。此操作不会影响生产环境。</Typography.Paragraph>
        <Popconfirm title="确认重置演示数据？" onConfirm={reset}><Button danger data-testid="reset-demo-button">重置演示数据</Button></Popconfirm>
      </Card>
    </>
  )
}

export default function App() {
  return (
    <AntApp>
      <Routes>
        <Route element={<EnterpriseLayout />}>
          <Route path="/oa" element={<OAListPage />} />
          <Route path="/oa/applications" element={<OAListPage />} />
          <Route path="/oa/applications/new" element={<OAFormPage />} />
          <Route path="/oa/applications/:id/edit" element={<OAFormPage />} />
          <Route path="/oa/applications/:id" element={<OADetailPage />} />
          <Route path="/oa/approvals" element={<OAApprovalWorkbench />} />
          <Route path="/oa/approvals/:id" element={<OAApprovalDetail />} />
          <Route path="/oa/new" element={<OAFormPage />} />
          <Route path="/oa/:id/edit" element={<OAFormPage />} />
          <Route path="/oa/:id" element={<OADetailPage />} />
          <Route path="/procurement" element={<ProcurementListPage />} />
          <Route path="/procurement/requests" element={<ProcurementListPage />} />
          <Route path="/procurement/new" element={<ProcurementPage />} />
          <Route path="/procurement/requests/:prNo" element={<ProcurementDetailPage />} />
          <Route path="/procurement/:prNo" element={<ProcurementDetailPage />} />
          <Route path="/erp" element={<Navigate to="/erp/workbench" replace />} />
          <Route path="/erp/workbench" element={<WorkbenchPage />} />
          <Route path="/erp/materials" element={<MaterialsPage />} />
          <Route path="/erp/po-candidates" element={<ERPPOCandidateListPage />} />
          <Route path="/erp/po-create/:taskId" element={<ERPPOCreatePage />} />
          <Route path="/erp/pos/:poNo" element={<ERPPODetailPage />} />
          <Route path="/erp/dashboard" element={<ERPBoardPage />} />
          <Route path="/erp/orders" element={<ERPOrderListPage />} />
          <Route path="/erp/orders/:poNo" element={<ERPPODetailPage />} />
          <Route path="/erp/requests/new" element={<ERPRequestNewPage />} />
          <Route path="/erp/export" element={<BatchExportPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="/" element={<Navigate to="/oa" replace />} />
          <Route path="*" element={<Navigate to="/oa" replace />} />
        </Route>
      </Routes>
    </AntApp>
  )
}
