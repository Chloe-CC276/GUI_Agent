import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Empty,
  Input,
  InputNumber,
  Space,
  Steps,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api'

type FormLine = {
  po_item_no?: number
  material_code?: string
  material_name?: string
  specification?: string
  quantity?: number | string
  uom?: string
  unit?: string
  unit_price_tax?: number | string
  unit_price?: number | string
  tax_rate?: number | string
  line_amount_tax?: number | string
  delivery_date?: string
}

type CreateContext = {
  task_id?: string | null
  pr_no: string
  status?: string
  po_no?: string | null
  steps?: Array<{ step_id: string; title?: string; status?: string }>
  form?: {
    header?: Record<string, unknown>
    lines?: FormLine[]
    oa_apply_no?: string
    award_confirmed_at?: string
    purchase_method?: string
  }
  source?: {
    oa_apply_no?: string
    award_confirmed_at?: string
    purchase_method?: string
  }
}

type VerifyError = { code?: string; message?: string }

const STARTABLE = new Set(['pending', 'QUEUED', 'WAITING', 'WAITING_PO', 'PREPARING'])

function buildPayload(header: Record<string, string>, lines: FormLine[]) {
  return {
    header,
    lines: lines.map((line, index) => ({
      ...line,
      po_item_no: line.po_item_no || index + 1,
      unit_price_tax: Number(line.unit_price_tax || line.unit_price || 0),
      quantity: Number(line.quantity || 0),
    })),
  }
}

export function ERPPOCreatePage() {
  const { taskId = '' } = useParams()
  const navigate = useNavigate()
  const [ctx, setCtx] = useState<CreateContext>()
  const [header, setHeader] = useState<Record<string, string>>({})
  const [lines, setLines] = useState<FormLine[]>([])
  const [loading, setLoading] = useState(true)
  const [savingDraft, setSavingDraft] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [creating, setCreating] = useState(false)
  const [createdPoNo, setCreatedPoNo] = useState<string>()
  const [verifyErrors, setVerifyErrors] = useState<VerifyError[]>([])
  const [verifyPassed, setVerifyPassed] = useState<boolean | null>(null)

  const applyContext = (data: CreateContext) => {
    setCtx(data)
    const h = (data.form?.header || {}) as Record<string, unknown>
    setHeader({
      supplier_code: String(h.supplier_code || ''),
      supplier_name: String(h.supplier_name || ''),
      request_dept: String(h.request_dept || ''),
      purchasing_org: String(h.purchasing_org || '1000'),
      purchasing_group: String(h.purchasing_group || 'P01'),
      currency_code: String(h.currency_code || 'CNY'),
      payment_terms: String(h.payment_terms || 'NET30'),
      buyer_id: String(h.buyer_id || 'BUYER-01'),
    })
    setLines((data.form?.lines || []).map((line) => ({ ...line })))
    if (data.po_no) setCreatedPoNo(data.po_no)
  }

  const load = async () => {
    setLoading(true)
    try {
      let data = await api.getPOCreateContext(taskId)
      if (data.task_id && STARTABLE.has(String(data.status || ''))) {
        await api.startPOTask(data.task_id)
        data = await api.getPOCreateContext(data.task_id)
      }
      applyContext(data)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [taskId])

  const total = useMemo(
    () => lines.reduce((sum, line) => sum + Number(line.quantity || 0) * Number(line.unit_price_tax || line.unit_price || 0), 0),
    [lines],
  )

  const currentStep = useMemo(() => {
    const steps = ctx?.steps || []
    const idx = steps.findIndex((step) => step.status !== 'success')
    if (createdPoNo) return steps.length
    return idx < 0 ? steps.length : idx
  }, [ctx, createdPoNo])

  const locked = Boolean(createdPoNo)

  const saveDraft = async () => {
    if (!ctx?.task_id) {
      message.error('缺少建单任务，请从待建 PO 列表发起')
      return
    }
    try {
      setSavingDraft(true)
      await api.savePODraft(ctx.task_id, buildPayload(header, lines))
      message.success('草稿已保存')
      setVerifyPassed(null)
      setVerifyErrors([])
      await load()
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSavingDraft(false)
    }
  }

  const reVerify = async () => {
    if (!ctx?.task_id) {
      message.error('缺少建单任务，请从待建 PO 列表发起')
      return
    }
    try {
      setVerifying(true)
      const result = await api.preSaveVerifyPO(ctx.task_id, buildPayload(header, lines))
      setVerifyPassed(Boolean(result.passed))
      setVerifyErrors(result.errors || [])
      if (result.passed) message.success('保存前校验通过')
      else message.warning('校验未通过，请修正后再创建')
      await load()
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setVerifying(false)
    }
  }

  const saveAndCreate = async () => {
    if (!ctx?.task_id) {
      message.error('缺少建单任务，请从待建 PO 列表发起')
      return
    }
    try {
      setCreating(true)
      const result = await api.createPOFromForm(ctx.task_id, buildPayload(header, lines))
      setCreatedPoNo(result.po_no)
      setVerifyPassed(true)
      setVerifyErrors([])
      message.success(`PO 创建成功：${result.po_no}`)
      // Keep po_no visible briefly so GUI Agent can verify READ_BACK before leaving.
      window.setTimeout(() => navigate(`/erp/pos/${result.po_no}`), 2800)
    } catch (e) {
      message.error((e as Error).message)
      await load()
    } finally {
      setCreating(false)
    }
  }

  if (loading) return <Typography.Text>加载 ERP 建单页…</Typography.Text>
  if (!ctx) return <Empty description="未找到建单上下文" />

  return (
    <div className="erp-classic-shell" data-testid="erp-po-create-page">
      <div className="page-title">
        <div>
          <Typography.Title level={3}>ERP 采购订单创建</Typography.Title>
          <Typography.Text type="secondary">
            草稿 → 校验 → 保存并创建 PO（ME21N）
          </Typography.Text>
        </div>
        <Space wrap>
          <Button onClick={() => navigate('/erp/po-candidates')}>返回待建列表</Button>
          <Button loading={savingDraft} disabled={locked} onClick={() => void saveDraft()} data-testid="erp-po-save-draft-button">
            保存草稿
          </Button>
          <Button loading={verifying} disabled={locked} onClick={() => void reVerify()} data-testid="erp-po-verify-button">
            重新校验
          </Button>
          <Button
            type="primary"
            danger
            loading={creating}
            disabled={locked}
            onClick={() => void saveAndCreate()}
            data-testid="erp-po-create-button"
          >
            保存并创建 PO
          </Button>
        </Space>
      </div>

      <Alert
        className="section-card"
        type="info"
        showIcon
        message="建单说明"
        description="「进入草稿」仅打开本页；「保存并创建 PO」才会落库并回写采购云。无采购云→ERP 建单业务 API。"
      />

      {verifyPassed === false ? (
        <Alert
          className="section-card"
          type="error"
          showIcon
          message="保存前校验失败"
          description={
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(verifyErrors.length ? verifyErrors : [{ message: '未知校验错误' }]).map((err, idx) => (
                <li key={`${err.code || 'err'}-${idx}`}>{err.code ? `[${err.code}] ` : ''}{err.message}</li>
              ))}
            </ul>
          }
        />
      ) : null}
      {verifyPassed === true && !createdPoNo ? (
        <Alert className="section-card" type="success" showIcon message="保存前校验通过，可继续保存并创建 PO" />
      ) : null}

      <Card title="执行步骤" className="section-card" data-testid="erp-po-steps">
        <Steps
          size="small"
          current={currentStep}
          items={(ctx.steps || []).map((step) => ({
            title: step.title || step.step_id,
            status: step.status === 'success' ? 'finish' : step.status === 'failed' ? 'error' : undefined,
          }))}
        />
      </Card>

      <Card title="来源信息（只读）" className="section-card erp-classic-card">
        <Descriptions column={3} size="small" bordered>
          <Descriptions.Item label="OA号">{ctx.source?.oa_apply_no || ctx.form?.oa_apply_no || '-'}</Descriptions.Item>
          <Descriptions.Item label="PR号">{ctx.pr_no}</Descriptions.Item>
          <Descriptions.Item label="定标时间">{String(ctx.source?.award_confirmed_at || ctx.form?.award_confirmed_at || '-')}</Descriptions.Item>
          <Descriptions.Item label="采购方式">{ctx.source?.purchase_method || ctx.form?.purchase_method || '-'}</Descriptions.Item>
          <Descriptions.Item label="任务状态"><Tag>{ctx.status || '-'}</Tag></Descriptions.Item>
          <Descriptions.Item label="task_id">{ctx.task_id || '-'}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="PO Header" className="section-card erp-classic-card" data-testid="erp-po-create-form">
        <div className="erp-form-grid">
          <label>供应商名称<input data-testid="erp-po-supplier" value={header.supplier_name || ''} onChange={(e) => setHeader({ ...header, supplier_name: e.target.value })} disabled={locked} /></label>
          <label>供应商编码<input data-testid="erp-po-supplier-code" value={header.supplier_code || ''} onChange={(e) => setHeader({ ...header, supplier_code: e.target.value })} disabled={locked} /></label>
          <label>采购组织<input data-testid="erp-po-purchasing-org" value={header.purchasing_org || ''} onChange={(e) => setHeader({ ...header, purchasing_org: e.target.value })} disabled={locked} /></label>
          <label>采购组<input data-testid="erp-po-purchasing-group" value={header.purchasing_group || ''} onChange={(e) => setHeader({ ...header, purchasing_group: e.target.value })} disabled={locked} /></label>
          <label>币种<input data-testid="erp-po-currency" value={header.currency_code || ''} onChange={(e) => setHeader({ ...header, currency_code: e.target.value })} disabled={locked} /></label>
          <label>付款条件<input data-testid="erp-po-payment-terms" value={header.payment_terms || ''} onChange={(e) => setHeader({ ...header, payment_terms: e.target.value })} disabled={locked} /></label>
          <label>需求部门<input data-testid="erp-po-request-dept" value={header.request_dept || ''} onChange={(e) => setHeader({ ...header, request_dept: e.target.value })} disabled={locked} /></label>
          <label>采购员<input data-testid="erp-po-buyer" value={header.buyer_id || ''} onChange={(e) => setHeader({ ...header, buyer_id: e.target.value })} disabled={locked} /></label>
        </div>
        <div className="erp-amount-bar">含税合计：¥{total.toFixed(2)}</div>
      </Card>

      <Card title="PO 物资行" className="section-card erp-classic-card">
        <Table
          rowKey={(_, index) => String(index)}
          pagination={false}
          dataSource={lines}
          columns={[
            {
              title: '行',
              render: (_, __, index) => <span data-testid="erp-po-line-row">{index + 1}</span>,
            },
            {
              title: '物料编码',
              render: (_, __, index) => (
                <Input
                  data-testid={`erp-po-line-material-${index}`}
                  value={lines[index]?.material_code || ''}
                  disabled={locked}
                  onChange={(e) => {
                    const next = [...lines]
                    next[index] = { ...next[index], material_code: e.target.value }
                    setLines(next)
                  }}
                />
              ),
            },
            { title: '物料名称', dataIndex: 'material_name' },
            { title: '规格', dataIndex: 'specification' },
            {
              title: '数量',
              render: (_, __, index) => (
                <InputNumber
                  data-testid={`erp-po-line-qty-${index}`}
                  min={0}
                  disabled={locked}
                  value={Number(lines[index]?.quantity || 0)}
                  onChange={(value) => {
                    const next = [...lines]
                    next[index] = { ...next[index], quantity: Number(value || 0) }
                    setLines(next)
                  }}
                />
              ),
            },
            { title: '单位', dataIndex: 'uom', render: (v, row) => v || row.unit },
            {
              title: '含税单价',
              render: (_, __, index) => (
                <InputNumber
                  data-testid={`erp-po-line-price-${index}`}
                  min={0}
                  disabled={locked}
                  value={Number(lines[index]?.unit_price_tax || lines[index]?.unit_price || 0)}
                  onChange={(value) => {
                    const next = [...lines]
                    next[index] = { ...next[index], unit_price_tax: Number(value || 0) }
                    setLines(next)
                  }}
                />
              ),
            },
            {
              title: '行金额',
              render: (_, row) => `¥${(Number(row.quantity || 0) * Number(row.unit_price_tax || row.unit_price || 0)).toFixed(2)}`,
            },
          ]}
        />
      </Card>

      {createdPoNo ? (
        <Alert
          className="section-card"
          type="success"
          showIcon
          message="PO 已创建"
          description={<span data-testid="erp-po-created-po-no">{createdPoNo}</span>}
        />
      ) : null}
    </div>
  )
}
