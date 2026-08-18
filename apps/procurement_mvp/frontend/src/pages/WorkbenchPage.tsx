import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Timeline,
  Typography,
  message,
} from 'antd'
import { useNavigate } from 'react-router-dom'
import { api, friendlyUnavailable, isApiUnavailable } from '../api'
import { WORKBENCH_STAGES } from '../config'
import type { StageState, WorkbenchStage, WorkbenchSummary, WorkflowEvent } from '../types'

function PageTitle({ title, subtitle, extra }: { title: string; subtitle: string; extra?: React.ReactNode }) {
  return (
    <div className="page-title">
      <div>
        <Typography.Title level={3} data-testid="workbench-title">{title}</Typography.Title>
        <Typography.Text type="secondary">{subtitle}</Typography.Text>
      </div>
      {extra}
    </div>
  )
}

const fallbackStages: WorkbenchStage[] = WORKBENCH_STAGES.map((name, index) => ({
  index: index + 1,
  name,
  state: (index === 0 ? 'current' : 'todo') as StageState,
}))

export function WorkbenchPage() {
  const navigate = useNavigate()
  const [summary, setSummary] = useState<WorkbenchSummary>()
  const [events, setEvents] = useState<WorkflowEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [unavailable, setUnavailable] = useState('')

  const load = async () => {
    setLoading(true)
    setUnavailable('')
    try {
      const data = await api.getWorkbenchSummary()
      setSummary(data)
      const key = data.current_pr?.request_no || data.current_pr?.oa_apply_no
      try {
        setEvents(await api.getWorkbenchEvents(key ? { business_key: key } : undefined))
      } catch (eventError) {
        if (isApiUnavailable(eventError)) {
          setEvents(data.recent_events || [])
        } else {
          message.error((eventError as Error).message)
        }
      }
    } catch (error) {
      if (isApiUnavailable(error)) {
        setUnavailable(friendlyUnavailable('工作台汇总', error))
        setSummary({
          stages: fallbackStages,
          metrics: { pending: 0, approving: 0, month_amount: 0, budget_rate: 0 },
          current_pr: null,
        })
        setEvents([])
      } else {
        message.error((error as Error).message)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { void load() }, [])

  if (loading) return <Spin data-testid="workbench-loading" />

  const stages: Array<WorkbenchStage & { status?: string; available?: boolean }> = summary?.stages?.length
    ? summary.stages
    : fallbackStages
  const metrics = summary?.metrics || {}
  const current = summary?.current_pr
  const normalizeState = (stage: WorkbenchStage & { status?: string; available?: boolean }, index: number): StageState => {
    const raw = stage.state || stage.status
    if (raw === 'completed' || raw === 'done') return 'done'
    if (raw === 'current') return 'current'
    if (raw === 'pending' || raw === 'todo') return stage.available === false ? 'unavailable' : 'todo'
    if (index + 1 === (summary?.current_stage || 1)) return 'current'
    if (index + 1 < (summary?.current_stage || 1)) return 'done'
    return 'todo'
  }

  return (
    <div data-testid="workbench-page">
      <PageTitle
        title="采购全流程工作台"
        subtitle="一眼查看阶段、待办、当前申请与流程动态"
        extra={
          <Space>
            <Button type="primary" onClick={() => navigate('/erp/requests/new')} data-testid="workbench-new-request">新建采购申请</Button>
            <Button onClick={() => navigate('/erp/export')} data-testid="workbench-export">批量导出与核对</Button>
            <Button onClick={load} data-testid="workbench-refresh">刷新</Button>
          </Space>
        }
      />
      {unavailable && <Alert type="warning" showIcon className="section-card" message={unavailable} data-testid="workbench-unavailable" />}
      <Card className="stage-bar-card" data-testid="workbench-stage-bar">
        <div className="stage-bar">
          {stages.map((stage, index) => {
            const state = normalizeState(stage, index)
            return (
              <button
                type="button"
                key={stage.name || index}
                className={`stage-item stage-${state}`}
                data-testid={`procurement-stage-${stage.index || index + 1}`}
                onClick={() => {
                  if (stage.route) navigate(stage.route)
                  else if (state === 'todo' || state === 'unavailable') message.info('该阶段暂未开放')
                }}
              >
                <span className="stage-index">{stage.index || index + 1}</span>
                <span className="stage-name">{stage.name}</span>
              </button>
            )
          })}
        </div>
      </Card>
      <Row gutter={16} className="section-card">
        <Col span={6}><Card hoverable data-testid="metric-pending" onClick={() => navigate('/procurement')}><Statistic title="待我处理" value={Number(metrics.pending || 0)} /></Card></Col>
        <Col span={6}><Card hoverable data-testid="metric-approving" onClick={() => navigate('/oa')}><Statistic title="审批中" value={Number(metrics.approving || 0)} /></Card></Col>
        <Col span={6}><Card data-testid="metric-month-amount"><Statistic title="本月采购额" prefix="¥" precision={2} value={Number(metrics.month_amount || 0)} /></Card></Col>
        <Col span={6}><Card data-testid="metric-budget-rate"><Statistic title="预算执行率" suffix="%" precision={1} value={Number(metrics.budget_rate || 0) * (Number(metrics.budget_rate || 0) <= 1 ? 100 : 1)} /></Card></Col>
      </Row>
      <Row gutter={16} className="section-card">
        <Col span={15}>
          <Card
            title="当前采购申请"
            data-testid="current-pr-card"
            extra={current && <Button type="link" onClick={() => navigate(`/procurement/${current.request_no}`)}>查看详情</Button>}
          >
            {!current ? (
              <Empty description="暂无当前采购申请">
                <Button type="primary" onClick={() => navigate('/erp/requests/new')}>新建采购申请</Button>
              </Empty>
            ) : (
              <>
                <Space wrap style={{ marginBottom: 12 }}>
                  <Tag color="blue">{current.request_no}</Tag>
                  <Tag>{current.status}</Tag>
                  <Typography.Text type="secondary">{current.oa_apply_no || '-'} · {current.oa_department || current.department || '-'}</Typography.Text>
                  <Typography.Text strong>合计 ¥{Number(current.total_amount || 0).toFixed(2)}</Typography.Text>
                </Space>
                <Table
                  size="small"
                  rowKey={(row) => String(row.id || row.line_no)}
                  pagination={false}
                  dataSource={(current.lines || []).slice(0, 8)}
                  columns={[
                    { title: '物料', dataIndex: 'material_name', render: (v, row) => v || row.raw_material_name || '-' },
                    { title: '编码', dataIndex: 'material_code', render: (v) => v || '-' },
                    { title: '数量', dataIndex: 'quantity' },
                    { title: '金额', dataIndex: 'line_amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
                  ]}
                />
              </>
            )}
          </Card>
        </Col>
        <Col span={9}>
          <Card title="流程动态" data-testid="workflow-timeline">
            {!events.length ? <Empty description="暂无流程事件" /> : (
              <Timeline
                items={events.slice(0, 12).map((event) => ({
                  children: (
                    <div>
                      <Typography.Text strong>{event.event_type}</Typography.Text>
                      <div><Typography.Text type="secondary">{event.event_time} · {event.operator || 'system'}</Typography.Text></div>
                      <div>{event.detail || event.status || ''}</div>
                    </div>
                  ),
                }))}
              />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
