import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Space,
  Spin,
  message,
} from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api'
import type { OARequest } from '../../types'
import { isOAStateConflict, isPendingApproval, normalizeOAStatus } from '../../utils/business'
import { ApprovalTimeline, PageTitle, StatusTag } from './shared'

export function OAApprovalDetail() {
  const { id = '' } = useParams()
  const navigate = useNavigate()
  const [record, setRecord] = useState<OARequest>()
  const [loading, setLoading] = useState(true)
  const [acting, setActing] = useState(false)
  const [opinion, setOpinion] = useState('')
  const [rejectForm] = Form.useForm<{ reason: string }>()

  const load = async () => {
    const data = await api.getOA(id)
    setRecord(data)
    setOpinion(data.approval_opinion || '')
    return data
  }

  useEffect(() => {
    load()
      .catch((e) => message.error(e.message))
      .finally(() => setLoading(false))
  }, [id])

  const handleConflict = (error: unknown) => {
    if (isOAStateConflict(error)) {
      message.error('状态已变化，请刷新')
      void load()
      return true
    }
    return false
  }

  const startApproval = () => {
    if (!record || acting) return
    Modal.confirm({
      title: '确认开始审批？',
      content: '开始后申请将锁定为审批中，申请人不可编辑。',
      okText: '开始审批',
      cancelText: '取消',
      okButtonProps: { 'data-testid': 'oa-start-approval-ok' } as React.ComponentProps<typeof Button>,
      onOk: async () => {
        try {
          setActing(true)
          const updated = await api.startOAApproval(record.id, { row_version: record.row_version || 1 })
          setRecord(updated)
          message.success('已开始审批')
        } catch (error) {
          if (!handleConflict(error)) message.error((error as Error).message)
          throw error
        } finally {
          setActing(false)
        }
      },
    })
  }

  const approve = () => {
    if (!record || acting) return
    Modal.confirm({
      title: '确认审批通过？',
      content: '通过后申请状态变为已通过，可被采购云承接。',
      okText: '审批通过',
      cancelText: '取消',
      okButtonProps: { 'data-testid': 'oa-approve-ok' } as React.ComponentProps<typeof Button>,
      onOk: async () => {
        try {
          setActing(true)
          const updated = await api.approveOA(record.id, {
            row_version: record.row_version || 1,
            opinion: opinion.trim() || undefined,
          })
          setRecord(updated)
          message.success('审批已通过')
        } catch (error) {
          if (!handleConflict(error)) message.error((error as Error).message)
          throw error
        } finally {
          setActing(false)
        }
      },
    })
  }

  const reject = () => {
    if (!record || acting) return
    rejectForm.resetFields()
    Modal.confirm({
      title: '驳回申请',
      content: (
        <Form form={rejectForm} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="reason"
            label="驳回原因"
            rules={[{ required: true, message: '请填写驳回原因' }]}
          >
            <Input.TextArea rows={4} data-testid="oa-reject-reason-input" />
          </Form.Item>
        </Form>
      ),
      okText: '确认驳回',
      okButtonProps: {
        danger: true,
        'data-testid': 'oa-reject-ok',
      } as React.ComponentProps<typeof Button>,
      onOk: async () => {
        const values = await rejectForm.validateFields()
        try {
          setActing(true)
          const updated = await api.rejectOA(record.id, {
            row_version: record.row_version || 1,
            reason: values.reason.trim(),
            opinion: values.reason.trim(),
          })
          setRecord(updated)
          message.success('已驳回')
        } catch (error) {
          if (!handleConflict(error)) message.error((error as Error).message)
          throw error
        } finally {
          setActing(false)
        }
      },
    })
  }

  if (loading) return <Spin data-testid="oa-approval-detail-loading" />
  if (!record) return <Empty description="未找到申请" />

  const status = normalizeOAStatus(record.status)
  const pendingStart = isPendingApproval(record)
  const inApproval = status === 'IN_APPROVAL'

  return (
    <div data-testid="oa-approval-detail-page">
      <PageTitle
        title={`审批详情 ${record.application_no}`}
        subtitle={pendingStart ? '待审批 · 等待审批人开始' : '人工审批操作页'}
        extra={
          <Space>
            <StatusTag status={record.status} isSubmitted={record.is_submitted} />
            {pendingStart ? <span data-testid="oa-pending-start-badge">待审批</span> : null}
            <Button onClick={() => navigate('/oa/approvals')} data-testid="oa-approval-back">
              返回工作台
            </Button>
          </Space>
        }
      />

      <Card title="申请信息" className="section-card" data-testid="oa-approval-header-card">
        <Descriptions column={3}>
          <Descriptions.Item label="标题">{record.title}</Descriptions.Item>
          <Descriptions.Item label="申请人">{record.applicant}</Descriptions.Item>
          <Descriptions.Item label="部门">{record.department}</Descriptions.Item>
          <Descriptions.Item label="预算金额">¥{Number(record.total_budget).toFixed(2)}</Descriptions.Item>
          <Descriptions.Item label="预算项目">
            {record.budget_project_name || record.budget_project_code || '-'}
          </Descriptions.Item>
          <Descriptions.Item label="紧急程度">{record.urgency_level || '-'}</Descriptions.Item>
          <Descriptions.Item label="提交时间">{record.submitted_at || '-'}</Descriptions.Item>
          <Descriptions.Item label="row_version">{record.row_version ?? 1}</Descriptions.Item>
          <Descriptions.Item label="采购原因" span={3}>
            {record.purchase_reason || '-'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card title="审批历史" className="section-card">
        <ApprovalTimeline history={record.approval_history} />
      </Card>

      {pendingStart && (
        <Card className="section-card" data-testid="oa-start-approval-card">
          <Alert
            type="info"
            showIcon
            message="该申请已提交，等待审批人开始审批"
            style={{ marginBottom: 16 }}
          />
          <Button
            type="primary"
            loading={acting}
            disabled={acting}
            onClick={startApproval}
            data-testid="oa-start-approval-button"
          >
            开始审批
          </Button>
        </Card>
      )}

      {inApproval && (
        <Card title="审批操作" className="section-card" data-testid="oa-approval-action-card">
          <Form layout="vertical">
            <Form.Item label="审批意见（可选）">
              <Input.TextArea
                rows={3}
                value={opinion}
                onChange={(e) => setOpinion(e.target.value)}
                data-testid="oa-approval-opinion-input"
              />
            </Form.Item>
          </Form>
          <Space>
            <Button
              type="primary"
              loading={acting}
              disabled={acting}
              onClick={approve}
              data-testid="oa-approve-button"
            >
              审批通过
            </Button>
            <Button
              danger
              loading={acting}
              disabled={acting}
              onClick={reject}
              data-testid="oa-reject-button"
            >
              驳回
            </Button>
          </Space>
        </Card>
      )}

      {!pendingStart && !inApproval && (
        <Alert
          className="section-card"
          type="success"
          showIcon
          message={`当前状态：${status}`}
          description={record.approval_opinion || '该申请已处理，仅可查看。'}
          data-testid="oa-approval-done-alert"
        />
      )}
    </div>
  )
}
