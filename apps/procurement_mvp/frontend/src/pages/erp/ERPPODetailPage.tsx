import { useEffect, useState } from 'react'
import { Button, Card, Descriptions, Drawer, Empty, Space, Table, Tag, Typography, message } from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api'

export function ERPPODetailPage() {
  const { poNo = '' } = useParams()
  const navigate = useNavigate()
  const [detail, setDetail] = useState<Awaited<ReturnType<typeof api.getPODetail>>>()
  const [lineage, setLineage] = useState<Awaited<ReturnType<typeof api.getPOLineage>>>()
  const [stepsOpen, setStepsOpen] = useState(false)
  const [steps, setSteps] = useState<Array<Record<string, unknown>>>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.getPODetail(poNo), api.getPOLineage(poNo)])
      .then(([d, l]) => {
        setDetail(d)
        setLineage(l)
      })
      .catch((e) => message.error((e as Error).message))
      .finally(() => setLoading(false))
  }, [poNo])

  const openSteps = async () => {
    const taskId = detail?.task_id
    if (!taskId) {
      message.info('无关联 Agent 任务')
      return
    }
    try {
      const data = await api.getAgentDashboardTaskSteps(taskId)
      setSteps(data.items || [])
      setStepsOpen(true)
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  if (loading) return <Typography.Text>加载中…</Typography.Text>
  if (!detail) return <Empty description="未找到 PO" />

  return (
    <>
      <div className="page-title">
        <div>
          <Typography.Title level={3}>PO 详情 {detail.po_no}</Typography.Title>
          <Typography.Text type="secondary">采购订单头、行与 OA-PR-PO 关系链</Typography.Text>
        </div>
        <Space>
          <Button onClick={() => navigate('/erp/po-candidates')}>返回列表</Button>
          <Button onClick={() => navigate(`/erp/dashboard?po_no=${detail.po_no}`)}>打开看板</Button>
          <Button type="primary" onClick={() => void openSteps()}>查看 Agent 轨迹</Button>
        </Space>
      </div>

      <Card title="PO 基础信息" data-testid="erp-po-detail-header">
        <Descriptions column={3} bordered size="small">
          <Descriptions.Item label="PO号">{detail.po_no}</Descriptions.Item>
          <Descriptions.Item label="状态"><Tag color="success">{detail.status}</Tag></Descriptions.Item>
          <Descriptions.Item label="创建时间">{String(detail.created_at || '-')}</Descriptions.Item>
          <Descriptions.Item label="供应商">{detail.supplier_name || detail.supplier_code || '-'}</Descriptions.Item>
          <Descriptions.Item label="采购组织">{detail.purchasing_org || '-'}</Descriptions.Item>
          <Descriptions.Item label="采购组">{detail.purchasing_group || '-'}</Descriptions.Item>
          <Descriptions.Item label="金额">¥{Number(detail.total_amount_tax || detail.total_amount || 0).toFixed(2)}</Descriptions.Item>
          <Descriptions.Item label="币种">{detail.currency_code || 'CNY'}</Descriptions.Item>
          <Descriptions.Item label="需求部门">{detail.request_dept || '-'}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="来源关系链" className="section-card" data-testid="erp-po-lineage">
        <Descriptions column={2} bordered size="small">
          <Descriptions.Item label="OA号">
            {detail.oa_apply_no ? (
              <Button type="link" onClick={() => navigate(`/oa/applications/${detail.oa_apply_no}`)}>{detail.oa_apply_no}</Button>
            ) : '-'}
          </Descriptions.Item>
          <Descriptions.Item label="PR号">
            <Button type="link" onClick={() => navigate(`/procurement/requests/${detail.pr_no}`)}>{detail.pr_no}</Button>
          </Descriptions.Item>
          <Descriptions.Item label="batch_id">{detail.batch_id || '-'}</Descriptions.Item>
          <Descriptions.Item label="task_id">{detail.task_id || '-'}</Descriptions.Item>
          <Descriptions.Item label="最新状态">{lineage?.latest_status || '-'}</Descriptions.Item>
          <Descriptions.Item label="task_ids">{(lineage?.task_ids || []).join(', ') || '-'}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="PO 物资行" className="section-card">
        <Table
          rowKey="id"
          pagination={false}
          dataSource={detail.lines || []}
          columns={[
            { title: '行号', dataIndex: 'po_item_no', render: (v, row) => v || row.line_no },
            { title: '物料编码', dataIndex: 'material_code' },
            { title: '名称', dataIndex: 'material_name' },
            { title: '规格', dataIndex: 'specification' },
            { title: '数量', dataIndex: 'quantity' },
            { title: '单位', dataIndex: 'uom', render: (v, row) => v || row.unit },
            { title: '单价', dataIndex: 'unit_price_tax', render: (v, row) => v ?? row.unit_price },
            { title: '税率', dataIndex: 'tax_rate', render: (v) => (v == null ? '-' : `${Number(v) * 100}%`) },
            { title: '行金额', dataIndex: 'line_amount_tax', render: (v, row) => `¥${Number(v ?? row.line_amount ?? 0).toFixed(2)}` },
          ]}
        />
      </Card>

      <Card title="Agent 执行摘要" className="section-card">
        <Descriptions column={3}>
          <Descriptions.Item label="执行状态">{detail.agent_summary?.status || '-'}</Descriptions.Item>
          <Descriptions.Item label="耗时(ms)">{detail.agent_summary?.duration_ms ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="重试次数">{detail.agent_summary?.retry_count ?? 0}</Descriptions.Item>
          <Descriptions.Item label="人工接管">{detail.agent_summary?.takeover_flag ? '是' : '否'}</Descriptions.Item>
          <Descriptions.Item label="执行器">{detail.agent_summary?.executor_type || '-'}</Descriptions.Item>
          <Descriptions.Item label="错误码">{detail.agent_summary?.error_code || '-'}</Descriptions.Item>
        </Descriptions>
      </Card>

      <Drawer title="Agent 步骤轨迹" open={stepsOpen} onClose={() => setStepsOpen(false)} width={560}>
        <Table
          rowKey={(row) => String(row.step_id)}
          dataSource={steps}
          pagination={false}
          columns={[
            { title: '步骤', dataIndex: 'step_name' },
            { title: '状态', dataIndex: 'status' },
            { title: '重试', dataIndex: 'retry_count' },
            { title: '错误', dataIndex: 'error_code', render: (v) => v || '-' },
          ]}
        />
      </Drawer>
    </>
  )
}
