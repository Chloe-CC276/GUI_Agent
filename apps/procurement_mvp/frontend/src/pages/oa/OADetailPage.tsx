import { useEffect, useState } from 'react'
import { Alert, Button, Card, Descriptions, Empty, Space, Spin, Table, message } from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api'
import type { Lineage, OALine, OARequest, Transfer } from '../../types'
import { canEditOA, canEnterProcurement, normalizeProcurementStatus } from '../../utils/business'
import { ApprovalTimeline, PageTitle, ProcurementStatusTag, StatusTag, nextTaskId } from './shared'

export function OADetailPage() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const [record, setRecord] = useState<OARequest>()
  const [lineage, setLineage] = useState<Lineage>()
  const [loading, setLoading] = useState(true)
  const [pushing, setPushing] = useState(false)

  const load = async () => {
    const data = await api.getOA(id)
    setRecord(data)
    try {
      setLineage(await api.getLineage(data.application_no))
    } catch {
      setLineage(undefined)
    }
  }

  useEffect(() => {
    load()
      .catch((e) => message.error(e.message))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <Spin data-testid="oa-detail-loading" />
  if (!record) return <Empty description="未找到申请" />

  const enabled = canEnterProcurement(record.status, record.procurement_status, record.linked_po_no)
  const editable = canEditOA(record.status, record.is_submitted)
  const prNo = record.linked_pr_no || lineage?.pr_no
  const procStatus = normalizeProcurementStatus(record.procurement_status) || 'NOT_STARTED'
  const viewOnly = Boolean(prNo) && (procStatus === 'AWARDED' || Boolean(record.linked_po_no) || !enabled)

  const enterProcurement = async () => {
    try {
      setPushing(true)
      let result: { pr_no?: string; redirect_url?: string }
      try {
        result = await api.submitProcurement(record.application_no, { task_id: nextTaskId('push-oa') })
      } catch (error) {
        const status = (error as { response?: { status?: number } })?.response?.status
        // Stale backend without procurement-cloud routes: fall back to legacy push.
        if (status === 404) {
          const legacy = await api.pushOA(record.application_no, { task_id: nextTaskId('push-oa') })
          result = {
            pr_no: legacy.pr_no,
            redirect_url: legacy.pr_no ? `/procurement/requests/${legacy.pr_no}` : undefined,
          }
          message.warning('后端缺少 submit-procurement，已回退旧推送接口；请重启 backend 以加载采购云增量')
        } else {
          throw error
        }
      }
      await load()
      const target = result.redirect_url || (result.pr_no ? `/procurement/requests/${result.pr_no}` : null)
      if (target) navigate(target)
      else message.warning('采购承接未生成 PR，请查看传输状态并重试')
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setPushing(false)
    }
  }

  const retry = async (transfer: Transfer) => {
    try {
      await api.retryTransfer(transfer.transfer_id, { task_id: nextTaskId('retry-transfer') })
      await load()
      message.success('传输已重试')
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  return (
    <div data-testid="oa-detail-page">
      <PageTitle
        title={`OA 申请 ${record.application_no}`}
        subtitle="申请内容与审批结果"
        extra={
          <Space>
            {editable ? (
              <Button onClick={() => navigate(`/oa/applications/${record.id}/edit`)} data-testid="oa-detail-edit-button">
                编辑
              </Button>
            ) : null}
            {viewOnly && prNo ? (
              <Button type="primary" onClick={() => navigate(`/procurement/requests/${prNo}`)} data-testid="enter-procurement-button">
                查看采购单
              </Button>
            ) : (
              <Button
                type="primary"
                disabled={!enabled}
                loading={pushing}
                onClick={enterProcurement}
                data-testid="enter-procurement-button"
              >
                提交采购
              </Button>
            )}
          </Space>
        }
      />
      {!enabled && !viewOnly && (
        <Alert
          type="warning"
          showIcon
          message="仅审批通过且未定标的 OA 申请可提交采购"
          data-testid="oa-gate-alert"
        />
      )}
      <Card title="申请头" data-testid="oa-header-card">
        <Descriptions column={3}>
          <Descriptions.Item label="标题">{record.title}</Descriptions.Item>
          <Descriptions.Item label="审批状态">
            <StatusTag status={record.status} isSubmitted={record.is_submitted} />
          </Descriptions.Item>
          <Descriptions.Item label="采购执行状态">
            <ProcurementStatusTag status={record.procurement_status} />
          </Descriptions.Item>
          <Descriptions.Item label="申请人">{record.applicant}</Descriptions.Item>
          <Descriptions.Item label="部门">{record.department}</Descriptions.Item>
          <Descriptions.Item label="申请时间">{record.created_at}</Descriptions.Item>
          <Descriptions.Item label="预算总额">¥{Number(record.total_budget).toFixed(2)}</Descriptions.Item>
          <Descriptions.Item label="预算项目">
            {record.budget_project_name || record.budget_project_code || '-'}
          </Descriptions.Item>
          <Descriptions.Item label="成本中心">{record.cost_center_code || '-'}</Descriptions.Item>
          <Descriptions.Item label="建议采买方式">{record.requested_method || '-'}</Descriptions.Item>
          <Descriptions.Item label="紧急程度">{record.urgency_level || '-'}</Descriptions.Item>
          <Descriptions.Item label="期望完成日">{record.expected_completion_date || '-'}</Descriptions.Item>
          <Descriptions.Item label="是否已提交">{record.is_submitted ? '是' : '否'}</Descriptions.Item>
          <Descriptions.Item label="审批通过时间">{record.approved_time || '-'}</Descriptions.Item>
          <Descriptions.Item label="审批人">{record.approved_by || record.current_approver_name || '-'}</Descriptions.Item>
          <Descriptions.Item label="关联PR">{record.linked_pr_no || '-'}</Descriptions.Item>
          <Descriptions.Item label="关联PO">{record.linked_po_no || '-'}</Descriptions.Item>
          <Descriptions.Item label="采购原因" span={3}>{record.purchase_reason || '-'}</Descriptions.Item>
          <Descriptions.Item label="备注" span={3}>{record.remark || '-'}</Descriptions.Item>
        </Descriptions>
      </Card>
      <Card title="物资明细" className="section-card" data-testid="oa-lines-card">
        <Table
          rowKey={(_, index) => String(index)}
          pagination={false}
          dataSource={record.lines}
          columns={[
            { title: '行号', render: (_: unknown, __: OALine, index: number) => index + 1 },
            { title: '需求物资', dataIndex: 'item_name' },
            { title: '规格', dataIndex: 'specification' },
            { title: '数量', dataIndex: 'quantity' },
            { title: '预估单价', dataIndex: 'estimated_unit_price' },
          ]}
        />
      </Card>
      <Card title="审批轨迹" className="section-card">
        <ApprovalTimeline history={record.approval_history} />
      </Card>
      {lineage ? (
        <Card title="跨系统来源链" className="section-card" data-testid="oa-lineage-card">
          <Descriptions column={3}>
            <Descriptions.Item label="OA">{lineage.oa_apply_no}</Descriptions.Item>
            <Descriptions.Item label="PR">{lineage.pr_no || '-'}</Descriptions.Item>
            <Descriptions.Item label="PO">{lineage.po_no || '-'}</Descriptions.Item>
          </Descriptions>
          <Table
            rowKey="transfer_id"
            pagination={false}
            dataSource={lineage.transfers || []}
            columns={[
              { title: '传输', dataIndex: 'transfer_type' },
              { title: '状态', dataIndex: 'status' },
              { title: '阶段', dataIndex: 'phase' },
              {
                title: '操作',
                render: (_: unknown, transfer: Transfer) =>
                  transfer.status !== 'success' ? (
                    <Button
                      type="link"
                      data-testid={`retry-transfer-${transfer.transfer_id}`}
                      onClick={() => void retry(transfer)}
                    >
                      重试
                    </Button>
                  ) : null,
              },
            ]}
          />
        </Card>
      ) : null}
    </div>
  )
}
