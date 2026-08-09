import { useEffect, useState } from 'react'
import { Button, Card, Empty, Input, Select, Space, Table, message } from 'antd'
import { PlusOutlined, SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { api } from '../../api'
import { oaStatusOptions, procurementStatusOptions } from '../../config'
import type { OARequest, PageResult } from '../../types'
import { canEditOA } from '../../utils/business'
import { PageTitle, ProcurementStatusTag, StatusTag } from './shared'

const DEFAULT_PAGE_SIZE = 15
const PAGE_SIZE_OPTIONS = ['10', '15', '20', '50', '100']

export function OAListPage() {
  const navigate = useNavigate()
  const [keyword, setKeyword] = useState('')
  const [status, setStatus] = useState<string>()
  const [procurementStatus, setProcurementStatus] = useState<string>()
  const [rows, setRows] = useState<OARequest[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [total, setTotal] = useState(0)

  const load = async (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true)
    try {
      const result: PageResult<OARequest> = await api.listOA({
        search: keyword || undefined,
        status,
        procurement_status: procurementStatus,
        page: nextPage,
        page_size: nextPageSize,
      })
      setRows(result.items || [])
      setTotal(result.pagination?.total ?? result.items?.length ?? 0)
      setPage(result.pagination?.page ?? nextPage)
      setPageSize(result.pagination?.page_size ?? nextPageSize)
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load(1, pageSize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, procurementStatus])

  return (
    <div data-testid="oa-list-page">
      <PageTitle
        title="OA 采购申请"
        subtitle="审批状态与采购执行状态分离监控：草稿/待审批/审批中/已通过/已驳回 + 未开始/采购准备中/已定标"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => navigate('/oa/applications/new')}
            data-testid="oa-create-button"
          >
            新建采购申请
          </Button>
        }
      />
      <Card>
        <Space wrap className="filter-bar">
          <Input
            allowClear
            prefix={<SearchOutlined />}
            placeholder="申请编号、标题、申请人"
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onPressEnter={() => void load(1, pageSize)}
            data-testid="oa-search-input"
          />
          <Select
            allowClear
            placeholder="审批状态"
            options={oaStatusOptions}
            value={status}
            onChange={setStatus}
            style={{ minWidth: 160 }}
            data-testid="oa-status-filter"
          />
          <Select
            allowClear
            placeholder="采购执行状态"
            options={procurementStatusOptions}
            value={procurementStatus}
            onChange={setProcurementStatus}
            style={{ minWidth: 160 }}
            data-testid="oa-procurement-status-filter"
          />
          <Button
            type="primary"
            onClick={() => void load(1, pageSize)}
            data-testid="oa-search-button"
          >
            查询
          </Button>
        </Space>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={rows}
          data-testid="oa-table"
          locale={{ emptyText: <Empty description="暂无 OA 申请" /> }}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: PAGE_SIZE_OPTIONS,
            showQuickJumper: true,
            showTotal: (count) => `共 ${count} 条`,
            onChange: (nextPage, nextPageSize) => {
              void load(nextPage, nextPageSize || pageSize)
            },
          }}
          columns={[
            { title: '申请编号', dataIndex: 'application_no' },
            { title: '申请标题', dataIndex: 'title' },
            { title: '申请人', dataIndex: 'applicant' },
            { title: '部门', dataIndex: 'department' },
            {
              title: '审批状态',
              dataIndex: 'status',
              render: (value, row) => (
                <StatusTag status={value} isSubmitted={row.is_submitted} />
              ),
            },
            {
              title: '采购执行状态',
              dataIndex: 'procurement_status',
              render: (value) => <ProcurementStatusTag status={value} />,
            },
            {
              title: '预算金额',
              dataIndex: 'total_budget',
              render: (value: number) => `¥${Number(value).toFixed(2)}`,
            },
            { title: '申请时间', dataIndex: 'created_at' },
            {
              title: '操作',
              render: (_, row) => {
                const editable = canEditOA(row.status, row.is_submitted)
                return (
                  <Space>
                    <Button
                      type="link"
                      onClick={() => navigate(`/oa/applications/${row.id}`)}
                      data-testid={`oa-view-${row.id}`}
                    >
                      查看详情
                    </Button>
                    {editable ? (
                      <Button
                        type="link"
                        onClick={() => navigate(`/oa/applications/${row.id}/edit`)}
                        data-testid={`oa-edit-${row.id}`}
                      >
                        编辑
                      </Button>
                    ) : null}
                  </Space>
                )
              },
            },
          ]}
        />
      </Card>
    </div>
  )
}
