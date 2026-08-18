import { useEffect, useState } from 'react'
import {
  Button,
  Card,
  Col,
  DatePicker,
  Input,
  Row,
  Space,
  Statistic,
  Table,
  Tabs,
  Typography,
  message,
} from 'antd'
import dayjs from 'dayjs'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api } from '../../api'

export function ERPBoardPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [department, setDepartment] = useState<string>()
  const [batchId, setBatchId] = useState(searchParams.get('batch_id') || undefined)
  const [supplier, setSupplier] = useState<string>()
  const [range, setRange] = useState<[dayjs.Dayjs | null, dayjs.Dayjs | null] | null>(null)

  const [agentSummary, setAgentSummary] = useState<Record<string, number>>({})
  const [funnel, setFunnel] = useState<Record<string, number>>({})
  const [events, setEvents] = useState<Array<Record<string, unknown>>>([])
  const [tasks, setTasks] = useState<Array<Record<string, unknown>>>([])
  const [poSummary, setPoSummary] = useState<Record<string, number | string>>({})
  const [trend, setTrend] = useState<Array<Record<string, unknown>>>([])
  const [byDept, setByDept] = useState<Array<Record<string, unknown>>>([])
  const [bySupplier, setBySupplier] = useState<Array<Record<string, unknown>>>([])
  const [recent, setRecent] = useState<Array<Record<string, unknown>>>([])

  const filters = () => ({
    department,
    batch_id: batchId,
    supplier,
    date_from: range?.[0]?.toISOString(),
    date_to: range?.[1]?.toISOString(),
  })

  const load = async () => {
    try {
      const f = filters()
      const [
        aSummary,
        aFunnel,
        aEvents,
        aTasks,
        pSummary,
        pTrend,
        pDept,
        pSupplier,
        pRecent,
      ] = await Promise.all([
        api.getAgentDashboardSummary({
          date_from: f.date_from,
          date_to: f.date_to,
          department: f.department,
          batch_id: f.batch_id,
        }),
        api.getAgentDashboardFunnel({
          date_from: f.date_from,
          date_to: f.date_to,
          batch_id: f.batch_id,
        }),
        api.getAgentDashboardEvents({ page_size: 20 }),
        api.getAgentDashboardTasks({ batch_id: f.batch_id }),
        api.getPODashboardSummary({
          date_from: f.date_from,
          date_to: f.date_to,
          department: f.department,
          supplier: f.supplier,
        }),
        api.getPODashboardTrend({ grain: 'day', date_from: f.date_from, date_to: f.date_to }),
        api.getPODashboardByDepartment(),
        api.getPODashboardBySupplier({ limit: 8 }),
        api.getPODashboardRecent({ page_size: 10 }),
      ])
      setAgentSummary(aSummary as Record<string, number>)
      setFunnel(aFunnel as Record<string, number>)
      setEvents(aEvents.items || [])
      setTasks(aTasks.items || [])
      setPoSummary(pSummary as Record<string, number | string>)
      setTrend(pTrend.items || [])
      setByDept(pDept.items || [])
      setBySupplier(pSupplier.items || [])
      setRecent(pRecent.items || [])
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const filterBar = (
    <Space wrap className="filter-bar">
      <DatePicker.RangePicker value={range as never} onChange={(value) => setRange(value)} />
      <Input
        placeholder="部门"
        value={department}
        onChange={(e) => setDepartment(e.target.value || undefined)}
        style={{ width: 140 }}
        allowClear
      />
      <Input
        placeholder="供应商"
        value={supplier}
        onChange={(e) => setSupplier(e.target.value || undefined)}
        style={{ width: 160 }}
        allowClear
      />
      <Input
        placeholder="batch_id"
        value={batchId}
        onChange={(e) => setBatchId(e.target.value || undefined)}
        style={{ width: 200 }}
        allowClear
      />
      <Button type="primary" onClick={() => void load()}>查询</Button>
    </Space>
  )

  return (
    <>
      <div className="page-title">
        <div>
          <Typography.Title level={3}>ERP 看板</Typography.Title>
          <Typography.Text type="secondary">Agent 执行安全 / PO 数据统计</Typography.Text>
        </div>
      </div>
      <Card>{filterBar}</Card>
      <Card className="section-card">
        <Tabs
          items={[
            {
              key: 'agent',
              label: 'Agent 执行与安全',
              children: (
                <>
                  <Row gutter={16}>
                    <Col span={6}><Card><Statistic title="待执行 PR" value={agentSummary.waiting_pr_count || 0} /></Card></Col>
                    <Col span={6}><Card><Statistic title="Agent 创建 PO" value={agentSummary.agent_created_po_count || 0} /></Card></Col>
                    <Col span={6}><Card><Statistic title="创建成功率" value={Number(agentSummary.success_rate || 0) * 100} precision={1} suffix="%" /></Card></Col>
                    <Col span={6}><Card><Statistic title="一次成功率" value={Number(agentSummary.first_pass_rate || 0) * 100} precision={1} suffix="%" /></Card></Col>
                  </Row>
                  <Row gutter={16} style={{ marginTop: 16 }}>
                    <Col span={6}><Card><Statistic title="平均耗时(s)" value={agentSummary.avg_duration_seconds || 0} precision={1} /></Card></Col>
                    <Col span={6}><Card><Statistic title="重试次数" value={agentSummary.retry_count_total || 0} /></Card></Col>
                    <Col span={6}><Card><Statistic title="人工接管" value={agentSummary.takeover_count || 0} /></Card></Col>
                    <Col span={6}><Card><Statistic title="重复拦截" value={agentSummary.duplicate_blocked_count || 0} /></Card></Col>
                  </Row>
                  <Card title="执行漏斗" className="section-card">
                    <Table
                      pagination={false}
                      rowKey="name"
                      dataSource={Object.entries(funnel).map(([name, value]) => ({ name, value }))}
                      columns={[
                        { title: '节点', dataIndex: 'name' },
                        { title: '数量', dataIndex: 'value' },
                      ]}
                    />
                  </Card>
                  <Card title="异常与安全记录" className="section-card">
                    <Table
                      rowKey={(row) => String(row.event_id)}
                      dataSource={events}
                      columns={[
                        { title: '时间', dataIndex: 'created_at', render: (v) => String(v || '-') },
                        { title: '类型', dataIndex: 'event_type' },
                        { title: '级别', dataIndex: 'severity' },
                        { title: '阶段', dataIndex: 'stage' },
                        { title: 'PR', dataIndex: 'pr_no' },
                        { title: '动作', dataIndex: 'action_taken' },
                      ]}
                    />
                  </Card>
                  <Card title="任务与批次" className="section-card">
                    <Table
                      rowKey={(row) => String(row.task_id)}
                      dataSource={tasks}
                      columns={[
                        { title: 'task_id', dataIndex: 'task_id' },
                        { title: 'PR', dataIndex: 'pr_no' },
                        { title: '状态', dataIndex: 'status' },
                        { title: 'PO', dataIndex: 'po_no', render: (v) => v || '-' },
                        { title: 'batch', dataIndex: 'batch_id' },
                        {
                          title: '操作',
                          render: (_, row) => (
                            <Typography.Link onClick={() => navigate(`/erp/po-create/${row.task_id}`)}>
                              打开
                            </Typography.Link>
                          ),
                        },
                      ]}
                    />
                  </Card>
                </>
              ),
            },
            {
              key: 'po',
              label: 'PO 数据统计',
              children: (
                <>
                  <Row gutter={16}>
                    <Col span={6}><Card><Statistic title="PO 总数" value={Number(poSummary.po_count || 0)} /></Card></Col>
                    <Col span={6}><Card><Statistic title="PO 总金额" prefix="¥" value={Number(poSummary.po_total_amount || 0)} precision={2} /></Card></Col>
                    <Col span={6}><Card><Statistic title="本月新增" value={Number(poSummary.month_new_po_count || 0)} /></Card></Col>
                    <Col span={6}><Card><Statistic title="关联完整率" value={Number(poSummary.lineage_complete_rate || 0) * 100} precision={1} suffix="%" /></Card></Col>
                  </Row>
                  <Card title="创建趋势" className="section-card">
                    <Table
                      rowKey={(row) => String(row.period)}
                      dataSource={trend}
                      pagination={false}
                      columns={[
                        { title: '周期', dataIndex: 'period' },
                        { title: 'PO数', dataIndex: 'po_count' },
                        { title: '金额', dataIndex: 'amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
                      ]}
                    />
                  </Card>
                  <Row gutter={16} className="section-card">
                    <Col span={12}>
                      <Card title="部门分布">
                        <Table
                          rowKey={(row) => String(row.department)}
                          dataSource={byDept}
                          pagination={false}
                          columns={[
                            { title: '部门', dataIndex: 'department' },
                            { title: 'PO数', dataIndex: 'po_count' },
                            { title: '金额', dataIndex: 'amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
                          ]}
                        />
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card title="供应商 Top">
                        <Table
                          rowKey={(row) => String(row.supplier)}
                          dataSource={bySupplier}
                          pagination={false}
                          columns={[
                            { title: '供应商', dataIndex: 'supplier' },
                            { title: 'PO数', dataIndex: 'po_count' },
                            { title: '金额', dataIndex: 'amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
                          ]}
                        />
                      </Card>
                    </Col>
                  </Row>
                  <Card title="最近 PO" className="section-card">
                    <Table
                      rowKey={(row) => String(row.po_no)}
                      dataSource={recent}
                      columns={[
                        { title: 'PO', dataIndex: 'po_no' },
                        { title: 'PR', dataIndex: 'pr_no' },
                        { title: 'OA', dataIndex: 'oa_apply_no' },
                        { title: '供应商', dataIndex: 'supplier_name' },
                        { title: '金额', dataIndex: 'total_amount', render: (v) => `¥${Number(v || 0).toFixed(2)}` },
                        {
                          title: '操作',
                          render: (_, row) => (
                            <Typography.Link onClick={() => navigate(`/erp/pos/${row.po_no}`)}>详情</Typography.Link>
                          ),
                        },
                      ]}
                    />
                  </Card>
                </>
              ),
            },
          ]}
        />
      </Card>
    </>
  )
}
