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
  { value: 'QUEUED', label: '已入队' },
  { value: 'RUNNING', label: '执行中' },
  { value: 'FAILED', label: '失败' },
  { value: 'WAIT_USER', label: '待人工' },
  { value: 'PO_CREATED', label: '已创建' },
]

function statusColor(status: string) {
  if (status === 'PO_CREATED') return 'success'
  if (status === 'FAILED') return 'error'
  if (status === 'WAIT_USER' || status === 'WAITING_PO') return 'warning'
  if (status === 'RUNNING' || status === 'QUEUED') return 'processing'
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
    () => rows.filter((row) => row.status === 'WAITING_PO' || row.status === 'FAILED' || row.status === 'QUEUED').map((row) => row.pr_no),
    [rows],
  )

  const createBatch = async (prNos: string[]) => {
    if (!prNos.length) {
      message.warning('请先勾选待建 PO 的单据')
      return
    }
    try {
      setCreating(true)
      const result = await api.createPOBatch({ pr_nos: prNos, operator: '采购员' })
      message.success(`已创建批次 ${result.batch_id}`)
      const firstRunnable = (result.tasks || []).find((item) => item.task_id && item.status === 'QUEUED')
      await load()
      if (firstRunnable?.task_id) {
        await api.runPOTask(firstRunnable.task_id)
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
          <Typography.Text type="secondary">仅展示采购云已定标且待写入 ERP 的 PR（方案 A：无采购云→ERP 业务 API）</Typography.Text>
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
            onClick={() => void createBatch(selected)}
            data-testid="erp-batch-create-po-button"
          >
            批量创建 PO
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
                  {(row.status === 'WAITING_PO' || row.status === 'FAILED' || row.status === 'QUEUED') && !row.po_no ? (
                    <Button
                      type="link"
                      data-testid={`erp-create-po-${row.pr_no}`}
                      onClick={() => void createBatch([row.pr_no])}
                    >
                      单条创建
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
