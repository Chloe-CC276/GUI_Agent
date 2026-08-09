import { Tag, Timeline, Typography } from 'antd'
import type { ApprovalHistory, OAStatusInput, ProcurementStatusInput } from '../../types'
import { getOAStatusMeta, getProcurementStatusMeta } from '../../utils/business'

export function PageTitle({ title, subtitle, extra }: { title: string; subtitle: string; extra?: React.ReactNode }) {
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

export function StatusTag({
  status,
  isSubmitted,
}: {
  status?: OAStatusInput | null
  isSubmitted?: boolean | null
}) {
  const meta = getOAStatusMeta(status, isSubmitted)
  return <Tag color={meta.color} data-testid="oa-status-tag">{meta.label}</Tag>
}

export function ProcurementStatusTag({ status }: { status?: ProcurementStatusInput | null }) {
  const meta = getProcurementStatusMeta(status)
  return <Tag color={meta.color} data-testid="procurement-status-tag">{meta.label}</Tag>
}

const actionLabel: Record<string, string> = {
  SUBMIT: '提交审批',
  START: '开始审批',
  APPROVE: '审批通过',
  REJECT: '驳回',
  RESUBMIT: '重新提交',
}

export function ApprovalTimeline({ history = [] }: { history?: ApprovalHistory[] }) {
  if (!history.length) {
    return <Typography.Text type="secondary" data-testid="approval-timeline-empty">暂无审批记录</Typography.Text>
  }
  return (
    <Timeline
      data-testid="approval-timeline"
      items={history.map((item) => ({
        children: (
          <div data-testid={`approval-history-${item.action}-${item.id || item.created_at}`}>
            <div>
              <strong>{actionLabel[item.action] || item.action}</strong>
              {item.operator_name ? ` · ${item.operator_name}` : ''}
            </div>
            <div>
              <Typography.Text type="secondary">{item.created_at}</Typography.Text>
              {item.from_status || item.to_status ? (
                <Typography.Text type="secondary">
                  {' '}
                  · {getOAStatusMeta(item.from_status).label} → {getOAStatusMeta(item.to_status).label}
                </Typography.Text>
              ) : null}
            </div>
            {item.opinion ? <div>{item.opinion}</div> : null}
          </div>
        ),
      }))}
    />
  )
}

export const nextTaskId = (() => {
  let seq = 0
  return (operation: string) => `web-${operation}-${Date.now()}-${++seq}`
})()
