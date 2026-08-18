import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  DatePicker,
  Form,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import dayjs from 'dayjs'
import { api, friendlyUnavailable, isApiUnavailable } from '../api'
import type { ExportCandidate, PurchaseMethodRules } from '../types'

const { RangePicker } = DatePicker

let taskSequence = 0
const nextTaskId = (operation: string) => `web-${operation}-${Date.now()}-${++taskSequence}`

function PageTitle({ title, subtitle, extra }: { title: string; subtitle: string; extra?: React.ReactNode }) {
  return (
    <div className="page-title">
      <div>
        <Typography.Title level={3}>{title}</Typography.Title>
        <Typography.Text type="secondary">{subtitle}</Typography.Text>
      </div>
      {extra}
    </div>
  )
}

const methodOptions = [
  { value: '网购', label: '网购' },
  { value: '比价', label: '比价' },
  { value: '招标', label: '招标' },
  { value: '集中采购', label: '集中采购' },
]

export function BatchExportPage() {
  const [form] = Form.useForm()
  const [rows, setRows] = useState<ExportCandidate[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [unavailable, setUnavailable] = useState('')
  const [rules, setRules] = useState<PurchaseMethodRules>()
  const [exporting, setExporting] = useState(false)

  const loadRules = async () => {
    try {
      setRules(await api.getPurchaseMethodRules())
    } catch (error) {
      if (isApiUnavailable(error)) {
        setRules({
          version: 'demo-fallback',
          rules: [
            { method: '网购', label: '网购', min_amount: 0, max_amount: 50000 },
            { method: '比价', label: '比价', min_amount: 50000, max_amount: 200000 },
            { method: '招标', label: '招标', min_amount: 200000, max_amount: 1000000 },
            { method: '集中采购', label: '集中采购', min_amount: 1000000, max_amount: null },
          ],
        })
      }
    }
  }

  const load = async () => {
    setLoading(true)
    setUnavailable('')
    const values = form.getFieldsValue()
    const statuses: string[] = values.status || []
    const params: Record<string, string | number | boolean | undefined> = {
      keyword: values.keyword || undefined,
      department: values.department || undefined,
      status: statuses.length === 1 ? statuses[0] : undefined,
      min_amount: values.min_amount,
      max_amount: values.max_amount,
      exportable_only: values.exportable_only || false,
    }
    try {
      const page = await api.listExportCandidates(params)
      const filtered = statuses.length > 1
        ? page.items.filter((item) => statuses.includes(item.status))
        : page.items
      setRows(filtered)
    } catch (error) {
      if (isApiUnavailable(error)) {
        setUnavailable(friendlyUnavailable('导出候选', error))
        try {
          const fallback = await api.listProcurements({ status: 'submitted' })
          setRows(fallback.items.map((item) => ({
            request_no: item.request_no,
            department: item.oa_department || item.department,
            applicant: item.oa_applicant || item.applicant,
            content_summary: item.oa_title || item.request_no,
            total_amount: item.total_amount,
            status: item.status,
            purchase_method_suggested: item.purchase_method_suggested,
            purchase_method_confirmed: item.purchase_method_confirmed,
            validation_status: item.export_status || 'review',
            exportable: item.status === 'submitted' || item.status === 'ready',
            oa_apply_no: item.oa_apply_no,
          })))
        } catch {
          setRows([])
        }
      } else {
        message.error((error as Error).message)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    form.setFieldsValue({
      date_range: [dayjs().startOf('year'), dayjs()],
      status: ['approved', 'submitted', 'ready'],
      exportable_only: false,
    })
    void loadRules()
    void load()
  }, [])

  const confirmMethod = async (prNo: string, method: string) => {
    try {
      await api.patchPurchaseMethod(prNo, { task_id: nextTaskId('confirm-method'), purchase_method_confirmed: method })
      setRows((current) => current.map((row) => row.request_no === prNo ? { ...row, purchase_method_confirmed: method } : row))
      message.success('采买方式已确认')
    } catch (error) {
      if (isApiUnavailable(error)) {
        setRows((current) => current.map((row) => row.request_no === prNo ? { ...row, purchase_method_confirmed: method } : row))
        message.warning(friendlyUnavailable('采买方式确认', error))
      } else message.error((error as Error).message)
    }
  }

  const validate = async () => {
    if (!selected.length) return message.warning('请先选择待导出申请')
    try {
      const result = await api.batchValidate({ pr_nos: selected })
      const map = new Map(result.items.map((item) => [item.request_no, item]))
      setRows((current) => current.map((row) => {
        const hit = map.get(row.request_no)
        if (!hit) return row
        return {
          ...row,
          validation_status: hit.validation_status,
          purchase_method_suggested: hit.purchase_method_suggested || row.purchase_method_suggested,
          purchase_method_confirmed: hit.purchase_method_confirmed || row.purchase_method_confirmed,
          exportable: hit.validation_status !== 'blocked',
        }
      }))
      message.success(`批量核对完成：通过 ${result.summary?.passed ?? 0} / 待核 ${result.summary?.review ?? 0} / 阻断 ${result.summary?.blocked ?? 0}`)
    } catch (error) {
      if (isApiUnavailable(error)) message.warning(friendlyUnavailable('批量核对', error))
      else message.error((error as Error).message)
    }
  }

  const doExport = async () => {
    if (!selected.length) return message.warning('请先选择待导出申请')
    const blocked = rows.filter((row) => selected.includes(row.request_no) && row.validation_status === 'blocked')
    if (blocked.length) return message.error(`存在阻断单据：${blocked.map((r) => r.request_no).join(', ')}`)
    Modal.confirm({
      title: '确认批量导出？',
      content: `将导出 ${selected.length} 张采购申请，属于高风险操作，请确认筛选与采买方式无误。`,
      okText: '确认导出',
      okButtonProps: { danger: true, 'data-testid': 'batch-export-confirm-ok' } as React.ComponentProps<typeof Button>,
      onOk: async () => {
        setExporting(true)
        try {
          const result = await api.batchExport({
            pr_nos: selected,
            template_version: 'V3.2',
            filters: { rule_version: rules?.version },
          })
          message.success(`导出任务已创建：${result.export_task_id}${result.file_name ? ` / ${result.file_name}` : ''}`)
          if (result.file_url) window.open(result.file_url, '_blank')
        } catch (error) {
          if (isApiUnavailable(error)) message.warning(friendlyUnavailable('批量导出', error))
          else message.error((error as Error).message)
        } finally {
          setExporting(false)
        }
      },
    })
  }

  const statusColor = (status?: string) => {
    if (status === 'pass' || status === 'passed' || status === 'ready') return 'success'
    if (status === 'blocked') return 'error'
    return 'warning'
  }

  return (
    <div data-testid="batch-export-page">
      <PageTitle title="批量申请导出与核对" subtitle="筛选、核对采买方式并显式确认后导出 Excel" />
      {unavailable && <Alert type="warning" showIcon message={unavailable} data-testid="export-unavailable" />}
      <Card>
        <Form form={form} layout="inline" className="filter-bar export-filter" onFinish={load}>
          <Form.Item name="date_range" label="申请日期"><RangePicker /></Form.Item>
          <Form.Item name="department" label="申请部门"><Input allowClear placeholder="部门" /></Form.Item>
          <Form.Item name="budget_project" label="预算项目"><Input allowClear placeholder="预算项目" /></Form.Item>
          <Form.Item name="status" label="采购状态"><Select mode="multiple" allowClear style={{ minWidth: 180 }} options={['draft', 'ready', 'submitted', 'approved'].map((v) => ({ value: v, label: v }))} /></Form.Item>
          <Form.Item name="min_amount" label="最小金额"><InputNumber min={0} /></Form.Item>
          <Form.Item name="max_amount" label="最大金额"><InputNumber min={0} /></Form.Item>
          <Form.Item name="keyword" label="关键字"><Input allowClear placeholder="申请号/物资/申请人" data-testid="export-keyword" /></Form.Item>
          <Form.Item name="exportable_only" label="仅可导出" valuePropName="checked"><Switch /></Form.Item>
          <Form.Item><Button type="primary" htmlType="submit" data-testid="export-search-button">查询候选</Button></Form.Item>
        </Form>
      </Card>
      <Card className="section-card" title="待导出采购申请" extra={
        <Space>
          <Button onClick={() => Modal.info({
            title: `采买方式规则 ${rules?.version || ''}`,
            content: <ul>{(rules?.rules || []).map((rule) => <li key={rule.method}>{rule.label}: {rule.min_amount} ~ {rule.max_amount ?? '∞'}</li>)}</ul>,
          })} data-testid="export-rules-button">规则配置</Button>
          <Button onClick={validate} data-testid="batch-validate-button">批量核对</Button>
          <Button type="primary" danger loading={exporting} onClick={doExport} data-testid="batch-export-button">确认导出</Button>
        </Space>
      }>
        <Table
          rowKey="request_no"
          loading={loading}
          dataSource={rows}
          data-testid="export-candidates-table"
          rowSelection={{
            selectedRowKeys: selected,
            onChange: (keys) => setSelected(keys as string[]),
            getCheckboxProps: (row) => ({ disabled: row.exportable === false }),
          }}
          columns={[
            { title: '申请编号', dataIndex: 'request_no' },
            { title: '申请部门', dataIndex: 'department', render: (v) => v || '-' },
            { title: '采购内容', dataIndex: 'content_summary', render: (v) => v || '-' },
            { title: '申请金额', dataIndex: 'total_amount', render: (v) => `¥${Number(v).toFixed(2)}` },
            { title: '系统建议', dataIndex: 'purchase_method_suggested', render: (v) => v || '-' },
            {
              title: '人工确认',
              render: (_, row) => (
                <Select
                  allowClear
                  placeholder="确认方式"
                  style={{ width: 120 }}
                  value={row.purchase_method_confirmed || undefined}
                  options={methodOptions}
                  onChange={(value) => value && confirmMethod(row.request_no, value)}
                  data-testid={`purchase-method-${row.request_no}`}
                />
              ),
            },
            {
              title: '校验状态',
              dataIndex: 'validation_status',
              render: (v, row) => <Tag color={statusColor(v)}>{v || '待核对'}{row.block_reason ? ` · ${row.block_reason}` : ''}</Tag>,
            },
          ]}
        />
      </Card>
    </div>
  )
}
