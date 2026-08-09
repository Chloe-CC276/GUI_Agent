import { useEffect, useState } from 'react'
import {
  Button,
  Card,
  Col,
  DatePicker,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Upload,
  message,
} from 'antd'
import type { UploadFile } from 'antd'
import { CloudUploadOutlined, DeleteOutlined, PlusOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../../api'
import { oaRequestedMethodOptions, oaUrgencyOptions } from '../../config'
import type { OAApplicationPayload, OARequest } from '../../types'
import { normalizeOAStatus } from '../../utils/business'
import { PageTitle, StatusTag } from './shared'

type LineValues = {
  item_name?: string
  specification?: string
  quantity?: number | null
  estimated_unit_price?: number | null
}

type FormValues = {
  title?: string
  department?: string
  applicant?: string
  budget_project_code?: string
  budget_project_name?: string
  cost_center_code?: string
  total_budget?: number
  purchase_reason?: string
  requested_method?: string
  urgency_level?: string
  expected_completion_date?: dayjs.Dayjs | null
  remark?: string
  lines?: LineValues[]
}

function lineAmount(line?: LineValues) {
  return Number(line?.quantity || 0) * Number(line?.estimated_unit_price || 0)
}

function calcTotalBudget(lines: LineValues[] = []) {
  return Number(
    lines.reduce((sum, line) => sum + lineAmount(line), 0).toFixed(2),
  )
}

function toPayload(values: FormValues, files: UploadFile[], rowVersion?: number): OAApplicationPayload {
  const lines = (values.lines || [])
    .filter((line) => (line.item_name || '').trim())
    .map((line) => ({
      item_name: (line.item_name || '').trim(),
      specification: (line.specification || '').trim() || undefined,
      quantity: Number(line.quantity || 0),
      estimated_unit_price: Number(line.estimated_unit_price || 0),
    }))
  const totalFromLines = calcTotalBudget(values.lines || [])
  return {
    title: values.title?.trim(),
    department: values.department?.trim(),
    applicant: values.applicant?.trim(),
    budget_project_code: values.budget_project_code?.trim(),
    budget_project_name: values.budget_project_name?.trim(),
    cost_center_code: values.cost_center_code?.trim(),
    total_budget:
      totalFromLines > 0
        ? totalFromLines
        : values.total_budget == null
          ? undefined
          : Number(values.total_budget),
    purchase_reason: values.purchase_reason?.trim(),
    requested_method: values.requested_method,
    urgency_level: values.urgency_level || 'NORMAL',
    expected_completion_date: values.expected_completion_date
      ? values.expected_completion_date.format('YYYY-MM-DD')
      : null,
    remark: values.remark?.trim(),
    lines,
    attachments: files.map((file) => ({
      file_name: file.name,
      file_url: (file.url as string) || `demo://${file.name}`,
    })),
    row_version: rowVersion,
  }
}

export function OAFormPage() {
  const navigate = useNavigate()
  const { id } = useParams()
  const isEdit = Boolean(id)
  const [form] = Form.useForm<FormValues>()
  const [record, setRecord] = useState<OARequest>()
  const [loading, setLoading] = useState(isEdit)
  const [saving, setSaving] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [files, setFiles] = useState<UploadFile[]>([])
  const [dirty, setDirty] = useState(false)
  const linesWatch = Form.useWatch('lines', form) || []

  useEffect(() => {
    const total = calcTotalBudget(linesWatch)
    if (total > 0 || (linesWatch.length > 0 && form.getFieldValue('total_budget') !== total)) {
      form.setFieldValue('total_budget', total)
    }
  }, [linesWatch, form])

  useEffect(() => {
    if (!id) return
    setLoading(true)
    api
      .getOA(id)
      .then((data) => {
        setRecord(data)
        form.setFieldsValue({
          title: data.title,
          department: data.department,
          applicant: data.applicant,
          budget_project_code: data.budget_project_code,
          budget_project_name: data.budget_project_name,
          cost_center_code: data.cost_center_code,
          total_budget: Number(data.total_budget || 0) || undefined,
          purchase_reason: data.purchase_reason,
          requested_method: data.requested_method,
          urgency_level: data.urgency_level || 'NORMAL',
          expected_completion_date: data.expected_completion_date ? dayjs(data.expected_completion_date) : null,
          remark: data.remark,
          lines:
            data.lines?.length
              ? data.lines.map((line) => ({
                  item_name: line.item_name,
                  specification: line.specification || '',
                  quantity: Number(line.quantity),
                  estimated_unit_price: Number(line.estimated_unit_price),
                }))
              : [{ item_name: '', specification: '', quantity: 1, estimated_unit_price: 0 }],
        })
        setFiles(
          (data.attachments || []).map((item, index) => ({
            uid: String(item.id ?? index),
            name: item.file_name,
            status: 'done',
            url: item.file_url,
          })),
        )
        setDirty(false)
      })
      .catch((error) => message.error((error as Error).message))
      .finally(() => setLoading(false))
  }, [id, form])

  const confirmLeave = () => {
    if (!dirty) {
      navigate('/oa')
      return
    }
    Modal.confirm({
      title: '确认取消？',
      content: '存在未保存的修改，离开后将丢失。',
      okText: '确认离开',
      cancelText: '继续编辑',
      okButtonProps: { 'data-testid': 'oa-form-leave-ok' } as React.ComponentProps<typeof Button>,
      onOk: () => navigate('/oa'),
    })
  }

  const saveDraft = async () => {
    try {
      const values = await form.validateFields(['title'])
      setSaving(true)
      const payload = toPayload({ ...form.getFieldsValue(true), ...values }, files, record?.row_version)
      if (!payload.title) {
        message.warning('请至少填写申请标题')
        return
      }
      if (isEdit && id) {
        const saved = await api.updateOA(id, payload)
        setRecord(saved)
        message.success('草稿已保存')
      } else {
        const created = await api.createOA(payload)
        message.success('草稿已保存')
        setDirty(false)
        navigate(`/oa/applications/${created.id}/edit`, { replace: true })
        return
      }
      setDirty(false)
    } catch (error) {
      if ((error as { errorFields?: unknown }).errorFields) return
      message.error((error as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const submitApproval = async () => {
    try {
      const values = await form.validateFields()
      const lines = values.lines || []
      if (!lines.some((line) => (line.item_name || '').trim() && Number(line.quantity) > 0)) {
        message.error('请至少填写一行物资明细（需求物资、数量）')
        return
      }
      setSubmitting(true)
      const payload = toPayload(values, files, record?.row_version)
      let current = record
      if (isEdit && id) {
        current = await api.updateOA(id, payload)
      } else {
        current = await api.createOA(payload)
      }
      const status = normalizeOAStatus(current.status)
      const actionPayload = { row_version: current.row_version || 1 }
      const result =
        status === 'REJECTED'
          ? await api.resubmitOA(current.id, actionPayload)
          : await api.submitOA(current.id, actionPayload)
      setDirty(false)
      message.success('已提交审批')
      navigate(`/oa/applications/${result.id}`)
    } catch (error) {
      if ((error as { errorFields?: unknown }).errorFields) return
      message.error((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) return <Spin data-testid="oa-form-loading" />

  return (
    <div data-testid="oa-form-page">
      <PageTitle
        title={isEdit ? '编辑采购申请' : '新建采购申请'}
        subtitle="填写申请主信息、物资明细、预算与采购原因后保存草稿或提交审批"
        extra={<StatusTag status={record?.status || 'DRAFT'} isSubmitted={record?.is_submitted} />}
      />
      <Form
        form={form}
        layout="vertical"
        initialValues={{
          urgency_level: 'NORMAL',
          applicant: '张三',
          department: '信息部',
          lines: [{ item_name: '', specification: '', quantity: 1, estimated_unit_price: 0 }],
        }}
        onValuesChange={() => setDirty(true)}
        data-testid="oa-application-form"
      >
        <Card title="申请基本信息" className="section-card" data-testid="oa-form-basic">
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="申请编号">
                <Input disabled value={record?.application_no || '保存后自动生成'} data-testid="oa-form-application-no" />
              </Form.Item>
            </Col>
            <Col span={16}>
              <Form.Item
                name="title"
                label="申请标题"
                rules={[
                  { required: true, message: '请填写申请标题' },
                  { min: 2, max: 100, message: '标题长度 2~100 字符' },
                ]}
              >
                <Input placeholder="请输入申请标题" data-testid="oa-form-title" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="department" label="申请部门" rules={[{ required: true, message: '请填写申请部门' }]}>
                <Input data-testid="oa-form-department" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="applicant" label="申请人" rules={[{ required: true, message: '请填写申请人' }]}>
                <Input data-testid="oa-form-applicant" />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        <Card title="物资明细" className="section-card" data-testid="oa-form-lines">
          <Form.List
            name="lines"
            rules={[
              {
                validator: async (_, lines: LineValues[]) => {
                  if (!lines?.length) throw new Error('请至少添加一行物资明细')
                },
              },
            ]}
          >
            {(fields, { add, remove }, { errors }) => (
              <>
                <Table
                  dataSource={fields}
                  pagination={false}
                  rowKey="key"
                  data-testid="oa-form-lines-table"
                  columns={[
                    {
                      title: '行号',
                      width: 70,
                      render: (_value, _field, index) => index + 1,
                    },
                    {
                      title: '需求物资',
                      render: (_value, field) => {
                        const { key, ...rest } = field
                        return (
                          <Form.Item
                            key={key}
                            {...rest}
                            name={[field.name, 'item_name']}
                            rules={[{ required: true, message: '请填写需求物资' }]}
                            style={{ marginBottom: 0 }}
                          >
                            <Input placeholder="物资名称" data-testid={`oa-line-item-name-${field.name}`} />
                          </Form.Item>
                        )
                      },
                    },
                    {
                      title: '规格',
                      render: (_value, field) => {
                        const { key, ...rest } = field
                        return (
                          <Form.Item key={key} {...rest} name={[field.name, 'specification']} style={{ marginBottom: 0 }}>
                            <Input placeholder="规格型号" data-testid={`oa-line-spec-${field.name}`} />
                          </Form.Item>
                        )
                      },
                    },
                    {
                      title: '数量',
                      width: 140,
                      render: (_value, field) => {
                        const { key, ...rest } = field
                        return (
                          <Form.Item
                            key={key}
                            {...rest}
                            name={[field.name, 'quantity']}
                            rules={[
                              { required: true, message: '请填写数量' },
                              { type: 'number', min: 0.0001, message: '数量须大于 0' },
                            ]}
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber
                              min={0.0001}
                              precision={4}
                              style={{ width: '100%' }}
                              data-testid={`oa-line-qty-${field.name}`}
                            />
                          </Form.Item>
                        )
                      },
                    },
                    {
                      title: '预估单价',
                      width: 150,
                      render: (_value, field) => {
                        const { key, ...rest } = field
                        return (
                          <Form.Item
                            key={key}
                            {...rest}
                            name={[field.name, 'estimated_unit_price']}
                            rules={[
                              { required: true, message: '请填写预估单价' },
                              { type: 'number', min: 0, message: '单价不能为负' },
                            ]}
                            style={{ marginBottom: 0 }}
                          >
                            <InputNumber
                              min={0}
                              precision={2}
                              style={{ width: '100%' }}
                              data-testid={`oa-line-price-${field.name}`}
                            />
                          </Form.Item>
                        )
                      },
                    },
                    {
                      title: '行金额',
                      width: 120,
                      render: (_value, field) => {
                        const line = linesWatch[field.name] || {}
                        return `¥${lineAmount(line).toFixed(2)}`
                      },
                    },
                    {
                      title: '操作',
                      width: 80,
                      render: (_value, field) => (
                        <Button
                          type="link"
                          danger
                          icon={<DeleteOutlined />}
                          disabled={fields.length <= 1}
                          onClick={() => remove(field.name)}
                          data-testid={`oa-line-remove-${field.name}`}
                        />
                      ),
                    },
                  ]}
                />
                <Space style={{ marginTop: 12 }}>
                  <Button
                    type="dashed"
                    icon={<PlusOutlined />}
                    onClick={() => add({ item_name: '', specification: '', quantity: 1, estimated_unit_price: 0 })}
                    data-testid="oa-line-add-button"
                  >
                    添加明细行
                  </Button>
                  <Form.ErrorList errors={errors} />
                </Space>
              </>
            )}
          </Form.List>
        </Card>

        <Card title="预算与采购信息" className="section-card" data-testid="oa-form-budget">
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="budget_project_name" label="预算项目名称" rules={[{ required: true, message: '请填写预算项目' }]}>
                <Input data-testid="oa-form-budget-project-name" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="budget_project_code" label="预算项目编码">
                <Input data-testid="oa-form-budget-project-code" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="cost_center_code" label="成本中心">
                <Input data-testid="oa-form-cost-center" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="total_budget"
                label="预算金额（按明细自动汇总）"
                rules={[
                  { required: true, message: '请填写预算金额' },
                  { type: 'number', min: 0.01, message: '预算金额须大于 0' },
                ]}
              >
                <InputNumber
                  min={0.01}
                  precision={2}
                  style={{ width: '100%' }}
                  data-testid="oa-form-total-budget"
                />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="requested_method" label="建议采买方式">
                <Select allowClear options={oaRequestedMethodOptions} data-testid="oa-form-requested-method" />
              </Form.Item>
            </Col>
            <Col span={24}>
              <Form.Item
                name="purchase_reason"
                label="采购原因"
                rules={[
                  { required: true, message: '请填写采购原因' },
                  { min: 10, message: '采购原因至少 10 个字符' },
                  { max: 1000, message: '采购原因不超过 1000 字符' },
                ]}
              >
                <Input.TextArea rows={4} data-testid="oa-form-purchase-reason" />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        <Card title="时间与紧急程度" className="section-card" data-testid="oa-form-schedule">
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="urgency_level" label="紧急程度" rules={[{ required: true, message: '请选择紧急程度' }]}>
                <Select options={oaUrgencyOptions} data-testid="oa-form-urgency" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="expected_completion_date" label="期望完成日期">
                <DatePicker style={{ width: '100%' }} data-testid="oa-form-expected-date" />
              </Form.Item>
            </Col>
          </Row>
        </Card>

        <Card title="附件" className="section-card" data-testid="oa-form-attachments">
          <Upload.Dragger
            fileList={files}
            multiple
            beforeUpload={() => false}
            onChange={({ fileList }) => {
              setFiles(fileList)
              setDirty(true)
            }}
            data-testid="oa-form-upload"
          >
            <CloudUploadOutlined className="upload-icon" />
            <p>附件仅生成 demo:// 引用，不上传二进制</p>
          </Upload.Dragger>
        </Card>

        <Card title="备注" className="section-card">
          <Form.Item name="remark" rules={[{ max: 1000, message: '备注不超过 1000 字符' }]}>
            <Input.TextArea rows={3} data-testid="oa-form-remark" />
          </Form.Item>
        </Card>

        <div className="form-actions">
          <Space>
            <Button onClick={confirmLeave} data-testid="oa-form-cancel-button">
              取消
            </Button>
            <Button loading={saving} onClick={saveDraft} data-testid="oa-save-draft-button">
              保存草稿
            </Button>
            <Button type="primary" loading={submitting} onClick={submitApproval} data-testid="oa-submit-approval-button">
              提交审批
            </Button>
          </Space>
        </div>
      </Form>
    </div>
  )
}
