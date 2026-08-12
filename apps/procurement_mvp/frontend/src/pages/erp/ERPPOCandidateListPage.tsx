import { useEffect, useMemo, useState } from 'react'
import { Button, Card, Empty, Input, Select, Space, Table, Tag, Typography, message } from 'antd'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api'

type Candidate = {
  pr_no: string
  oa_apply_no?: string
  title?: string
  department?: string
  supplier_code?: string
  supplier_name?: string
  line_count?: number
  total_amount?: number | string
  purchase_method?: string
  award_confirmed_at?: string
  status: string
  task_id?: string | null
  batch_id?: string | null
  po_no?: string | null
  error_code?: string | null
}

const statusOptions = [
  { value: 'WAITING_PO', label: '待建 PO' },
  { value: 'pending', label: '已入队' },
  { value: 'DRAFT_EDITING', label: '草稿编辑' },
  { value: 'PRE_SAVE_VERIFY', label: '保存前校验' },
  { value: 'FAILED', label: '失败' },
  { value: 'WAIT_USER', label: '待人工' },
  { value: 'SUCCESS', label: '已创建' },
  { value: 'DUPLICATE_BLOCKED', label: '重复拦截' },
]

const ENTER_DRAFT_STATUSES = new Set(['WAITING_PO', 'WAITING', 'FAILED', 'pending', 'QUEUED', 'DRAFT_EDITING'])

function statusColor(status: string) {
  if (status === 'SUCCESS' || status === 'PO_CREATED') return 'success'
  if (status === 'FAILED' || status === 'DUPLICATE_BLOCKED') return 'error'
  if (status === 'WAIT_USER' || status === 'WAITING_PO' || status === 'WAITING') return 'warning'
  if (status === 'DRAFT_EDITING' || status === 'PRE_SAVE_VERIFY' || status === 'pending' || status === 'QUEUED') {
    return 'processing'
  }
  return 'default'
}

export function ERPPOCandidateListPage() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<string | undefined>('WAITING_PO')
  const [q, setQ] = useState('')
  const [rows, setRows] = useState<Candidate[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [creating, setCreating] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await api.listPOCandidates({ status, q: q || undefined })
      setRows(data.items || [])
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [status])

  const selectableKeys = useMemo(
    () => rows.filter((row) => ENTER_DRAFT_STATUSES.has(row.status) && !row.po_no).map((row) => row.pr_no),
    [rows],
  )

  const enterDraft = async (prNos: string[]) => {
    if (!prNos.length) {
      message.warning('请先勾选待建 PO 的单据')
      return
    }
    try {
      setCreating(true)
      const result = await api.createPOBatch({ pr_nos: prNos, operator: '采购员' })
      message.success(`已创建批次 ${result.batch_id}（进入草稿，未创建 PO）`)
      const firstRunnable = (result.tasks || []).find((item) => item.task_id && !item.po_no)
      await load()
      if (firstRunnable?.task_id) {
        await api.startPOTask(firstRunnable.task_id)
        navigate(`/erp/po-create/${firstRunnable.task_id}`)
      }
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setCreating(false)
    }
  }

  return (
    <>
      <div className="page-title">
        <div>
          <Typography.Title level={3}>待创建 PO 列表</Typography.Title>
          <Typography.Text type="secondary">
            进入草稿后在 ERP 建单页完成创建与回写；无采购云→ERP 建单业务 API。
          </Typography.Text>
        </div>
      </div>
      <Card>
        <Space wrap className="filter-bar">
          <Select
            allowClear
            placeholder="状态筛选"
            value={status}
            onChange={(value) => setStatus(value)}
            options={statusOptions}
            style={{ minWidth: 160 }}
            data-testid="erp-po-status-filter"
          />
          <Input
            placeholder="PR / OA / 标题 / 供应商"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onPressEnter={() => void load()}
            allowClear
          />
          <Button type="primary" onClick={() => void load()}>查询</Button>
          <Button
            type="primary"
            danger
            loading={creating}
            disabled={!selected.length}
            onClick={() => void enterDraft(selected)}
            data-testid="erp-batch-create-po-button"
          >
            批量进入草稿
          </Button>
        </Space>
        <Table
          rowKey="pr_no"
          loading={loading}
          dataSource={rows}
          data-testid="erp-po-pending-table"
          locale={{ emptyText: <Empty description="暂无待建 PO" /> }}
          rowSelection={{
            selectedRowKeys: selected,
            onChange: (keys) => setSelected(keys as string[]),
            getCheckboxProps: (row) => ({
              disabled: !selectableKeys.includes(row.pr_no) || Boolean(row.po_no),
            }),
          }}
          columns={[
            { title: 'PR号', dataIndex: 'pr_no' },
            { title: 'OA号', dataIndex: 'oa_apply_no', render: (v) => v || '-' },
            { title: '标题', dataIndex: 'title', render: (v) => v || '-' },
            { title: '部门', dataIndex: 'department', render: (v) => v || '-' },
            { title: '供应商', render: (_, row) => row.supplier_name || row.supplier_code || '-' },
            { title: '行数', dataIndex: 'line_count' },
            { title: '金额', dataIndex: 'total_amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
            { title: '采购方式', dataIndex: 'purchase_method', render: (v) => v || '-' },
            {
              title: '状态',
              dataIndex: 'status',
              render: (value) => <Tag color={statusColor(value)}>{value}</Tag>,
            },
            {
              title: '操作',
              render: (_, row) => (
                <Space>
                  {ENTER_DRAFT_STATUSES.has(row.status) && !row.po_no ? (
                    <Button
                      type="link"
                      data-testid={`erp-create-po-${row.pr_no}`}
                      onClick={() => void enterDraft([row.pr_no])}
                    >
                      进入草稿
                    </Button>
                  ) : null}
                  {row.task_id ? (
                    <Button type="link" onClick={() => navigate(`/erp/po-create/${row.task_id}`)}>
                      打开建单
                    </Button>
                  ) : null}
                  {row.po_no ? (
                    <Button type="link" onClick={() => navigate(`/erp/pos/${row.po_no}`)}>
                      查看 PO
                    </Button>
                  ) : null}
                </Space>
              ),
            },
          ]}
        />
      </Card>
    </>
  )
}
