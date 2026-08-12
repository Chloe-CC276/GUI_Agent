import { useEffect, useState } from 'react'
import { Badge, Button, Card, Empty, Input, Space, Table, Tabs, message } from 'antd'
import { Search } from 'lucide-react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../../api'
import type { ApprovalQueue, OARequest, OAStatusInput } from '../../types'
import { isPendingApproval, normalizeOAStatus } from '../../utils/business'
import { PageTitle, StatusTag } from './shared'

const queueTabs: Array<{ key: ApprovalQueue; label: string }> = [
  { key: 'pending_start', label: '待审批' },
  { key: 'in_approval', label: '审批中' },
  { key: 'done', label: '已处理' },
]

export function OAApprovalWorkbench() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const initialQueue = (searchParams.get('queue') as ApprovalQueue) || 'pending_start'
  const [queue, setQueue] = useState<ApprovalQueue>(
    queueTabs.some((item) => item.key === initialQueue) ? initialQueue : 'pending_start',
  )
  const [keyword, setKeyword] = useState('')
  const [rows, setRows] = useState<OARequest[]>([])
  const [counts, setCounts] = useState<Record<ApprovalQueue, number>>({
    pending_start: 0,
    in_approval: 0,
    done: 0,
  })
  const [loading, setLoading] = useState(false)

  const loadCounts = async () => {
    const [pending, inApproval, done] = await Promise.all([
      api.listOAApprovals({ queue: 'pending_start' }),
      api.listOAApprovals({ queue: 'in_approval' }),
      api.listOAApprovals({ queue: 'done' }),
    ])
    setCounts({
      pending_start: pending.items?.length || 0,
      in_approval: inApproval.items?.length || 0,
      done: done.items?.length || 0,
    })
  }

  const load = async (nextQueue = queue) => {
    setLoading(true)
    try {
      const data = await api.listOAApprovals({ queue: nextQueue, search: keyword || undefined })
      setRows(data.items || [])
      await loadCounts()
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(queue)
  }, [queue])

  const onTabChange = (key: string) => {
    const next = key as ApprovalQueue
    setQueue(next)
    setSearchParams({ queue: next })
  }

  return (
    <div data-testid="oa-approval-workbench">
      <PageTitle
        title="OA 审批工作台"
        subtitle="与申请列表状态同步：待审批、审批中、已处理（通过/驳回及采购后续状态）"
      />
      <Card>
        <Tabs
          activeKey={queue}
          onChange={onTabChange}
          data-testid="oa-approval-tabs"
          items={queueTabs.map((tab) => ({
            key: tab.key,
            label: (
              <span data-testid={`oa-approval-tab-${tab.key}`}>
                <Badge
                  count={counts[tab.key]}
                  size="small"
                  offset={[8, -2]}
                  color={tab.key === 'in_approval' ? '#1677ff' : undefined}
                >
                  {tab.label}
                </Badge>
              </span>
            ),
          }))}
        />
        <Space wrap className="filter-bar">
          <Input
            allowClear
            prefix={<Search size={16} strokeWidth={1.75} />}
            placeholder="申请编号、标题、申请人、部门"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onPressEnter={() => load()}
            data-testid="oa-approval-search-input"
          />
          <Button type="primary" onClick={() => load()} data-testid="oa-approval-search-button">
            查询
          </Button>
        </Space>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={rows}
          data-testid="oa-approval-table"
          locale={{ emptyText: <Empty description="暂无审批任务" /> }}
          columns={[
            { title: '申请编号', dataIndex: 'application_no' },
            { title: '标题', dataIndex: 'title' },
            { title: '申请人', dataIndex: 'applicant' },
            { title: '部门', dataIndex: 'department' },
            {
              title: '预算金额',
              dataIndex: 'total_budget',
              render: (value: number) => `¥${Number(value).toFixed(2)}`,
            },
            {
              title: '提交时间',
              dataIndex: 'submitted_at',
              render: (value?: string) => value || '-',
            },
            {
              title: '状态',
              dataIndex: 'status',
              render: (value: OAStatusInput, row) => (
                <StatusTag status={value} isSubmitted={row.is_submitted} />
              ),
            },
            {
              title: '操作',
              render: (_, row) => {
                const status = normalizeOAStatus(row.status)
                const label = isPendingApproval(row)
                  ? '开始审批'
                  : status === 'IN_APPROVAL'
                    ? '继续审批'
                    : '查看详情'
                return (
                  <Button
                    type="link"
                    onClick={() => navigate(`/oa/approvals/${row.id}`)}
                    data-testid={`oa-approval-open-${row.id}`}
                  >
                    {label}
                  </Button>
                )
              },
            },
          ]}
        />
      </Card>
    </div>
  )
}
