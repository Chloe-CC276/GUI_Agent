import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  DatePicker,
  Divider,
  Form,
  Input,
  InputNumber,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Typography,
  Upload,
  message,
} from 'antd'
import type { UploadFile } from 'antd'
import { CloudUpload, Plus, Trash2 } from 'lucide-react'
import dayjs from 'dayjs'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, friendlyUnavailable, isApiUnavailable } from '../api'
import { purchaseTypeOptions } from '../config'
import type { AttachmentPayload, ImportBatch, Material, OARequest, ProcurementLine, ProcurementRequest } from '../types'
import { canEnterProcurement, getOAStatusMeta, lineAmount, totalAmount } from '../utils/business'

let taskSequence = 0
const nextTaskId = (operation: string) => `web-${operation}-${Date.now()}-${++taskSequence}`

function PageTitle({ title, subtitle, extra }: { title: string; subtitle: string; extra?: React.ReactNode }) {
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

export function ERPRequestNewPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const oaId = searchParams.get('oa_id')
  const [form] = Form.useForm()
  const [oa, setOA] = useState<OARequest>()
  const [oaOptions, setOAOptions] = useState<OARequest[]>([])
  const [oaLoading, setOALoading] = useState(Boolean(oaId))
  const [materials, setMaterials] = useState<Material[]>([])
  const [lines, setLines] = useState<ProcurementLine[]>([{ line_no: 1, quantity: 1 }])
  const [files, setFiles] = useState<UploadFile[]>([])
  const [request, setRequest] = useState<ProcurementRequest>()
  const [submitting, setSubmitting] = useState(false)
  const [importBatch, setImportBatch] = useState<ImportBatch>()
  const [importOpen, setImportOpen] = useState(false)
  const [importHint, setImportHint] = useState('')

  useEffect(() => {
    api.listOA({ status: 'approved' }).then((page) => setOAOptions(page.items)).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!oaId) {
      setOALoading(false)
      return
    }
    api.getOA(oaId).then((data) => {
      setOA(data)
      form.setFieldsValue({
        department: data.department,
        applicant: data.applicant,
        purchase_reason: data.title,
        receive_address: '总部收货仓',
        purchase_type: 'inquiry',
        expected_delivery_date: dayjs().add(14, 'day'),
      })
      if (data.lines.length) {
        setLines(data.lines.map((line, index) => ({
          line_no: index + 1,
          oa_line_id: line.id,
          material_name: line.item_name,
          specification: line.specification,
          quantity: line.quantity,
          unit_price: line.estimated_unit_price,
          raw_material_name: line.item_name,
          raw_specification: line.specification,
        })))
      }
    }).catch((e) => message.error(e.message)).finally(() => setOALoading(false))
  }, [oaId, form])

  const amount = useMemo(() => totalAmount(lines), [lines])
  const updateLine = (lineNo: number, patch: Partial<ProcurementLine>) =>
    setLines((current) => current.map((line) => line.line_no === lineNo ? { ...line, ...patch } : line))
  const addLine = () => setLines((current) => [...current, { line_no: Math.max(0, ...current.map((x) => x.line_no)) + 1, quantity: 1 }])
  const removeLine = (lineNo: number) => setLines((current) => current.filter((line) => line.line_no !== lineNo))

  const searchMaterials = async (search: string) => {
    try { setMaterials((await api.listMaterials({ search, page_size: 100 })).items.filter((item) => item.status === 'active')) }
    catch (e) { message.error((e as Error).message) }
  }

  const attachments = (): AttachmentPayload[] => files.map((file) => ({
    file_name: file.name,
    file_url: `demo://${encodeURIComponent(file.name)}`,
  }))

  const headerValues = async () => {
    const values = await form.validateFields()
    return {
      department: values.department as string,
      applicant: values.applicant as string,
      budget_project: values.budget_project as string,
      cost_center: values.cost_center as string,
      purchase_type: values.purchase_type as string,
      expected_delivery_date: values.expected_delivery_date ? dayjs(values.expected_delivery_date).format('YYYY-MM-DD') : undefined,
      receive_address: values.receive_address as string,
      purchase_reason: values.purchase_reason as string,
    }
  }

  const buildPayload = async (operation: string) => {
    const header = await headerValues()
    const source = oa
    if (!source) throw new Error('请先选择已审批通过的 OA 申请，或等待后端开放无 OA 建单接口')
    if (!canEnterProcurement(source.status)) throw new Error('必须从已审批通过的 OA 申请创建采购单')
    if (!lines.length || lines.some((line) => !line.material_code || !line.quantity || line.unit_price == null)) {
      throw new Error('请为每行匹配有效 ERP 物料，并填写数量和单价')
    }
    return {
      task_id: nextTaskId(operation),
      business_key: source.application_no,
      oa_application_id: source.id,
      ...header,
      lines: lines.map((line) => {
        const { line_no: _, ...payloadLine } = line
        return {
          ...payloadLine,
          material_code: line.material_code!,
          line_amount: lineAmount(line),
        }
      }),
      attachments: attachments(),
    }
  }

  const persistDraft = async (operation: string) => {
    const payload = await buildPayload(operation)
    let saved: ProcurementRequest
    if (request) {
      const { oa_application_id: _, ...updatePayload } = payload
      saved = await api.updateProcurementDraft(request.id, updatePayload)
    } else {
      saved = await api.createProcurement(payload)
    }
    setRequest(saved)
    return saved
  }

  const saveDraft = async () => {
    try {
      const saved = await persistDraft(request ? 'update-draft' : 'create-draft')
      message.success(`草稿已保存：${saved.request_no}`)
    } catch (e) { message.error((e as Error).message) }
  }

  const submit = async () => {
    try {
      setSubmitting(true)
      const saved = await persistDraft(request ? 'sync-draft' : 'create-before-submit')
      const businessKey = oa!.application_no
      const prepared = await api.prepareSubmit(saved.id, { task_id: nextTaskId('prepare-submit'), business_key: businessKey })
      if (!prepared.confirmation_token) throw new Error('prepare-submit 未返回 confirmation_token')
      const confirmed = await api.confirmSubmit(saved.id, {
        task_id: nextTaskId('confirm-submit'),
        business_key: businessKey,
        confirmation_token: prepared.confirmation_token,
      })
      setRequest(confirmed)
      message.success(`采购申请已提交：${confirmed.request_no}`)
      navigate(`/procurement/${confirmed.request_no}`)
    } catch (e) { message.error((e as Error).message) }
    finally { setSubmitting(false) }
  }

  const confirmSubmit = () => Modal.confirm({
    title: '确认提交采购申请？',
    content: '提交前将完成校验与显式确认，属于高风险操作。',
    okText: '确认并提交',
    cancelText: '取消',
    okButtonProps: { danger: true, 'data-testid': 'erp-submit-confirm-ok' } as React.ComponentProps<typeof Button>,
    onOk: submit,
  })

  const onImportUpload = async (file: File) => {
    setImportHint('')
    try {
      const preview = await api.previewImport(file)
      setImportBatch(preview)
      setImportOpen(true)
    } catch (error) {
      if (isApiUnavailable(error)) {
        setImportHint(friendlyUnavailable('Excel 导入', error))
        message.warning(friendlyUnavailable('Excel 导入', error))
      } else message.error((error as Error).message)
    }
    return false
  }

  const confirmImport = async () => {
    if (!importBatch) return
    try {
      const result = await api.confirmImport({
        task_id: nextTaskId('import-confirm'),
        business_key: oa?.application_no || request?.request_no || importBatch.import_batch_id,
        import_batch_id: importBatch.import_batch_id,
        pr_no: request?.request_no,
        oa_application_id: oa?.id,
        row_nos: importBatch.rows?.filter((row) => row.status === 'ok').map((row) => row.row_no),
      })
      const pr = result.request
      setRequest(pr)
      setLines(pr.lines.map((line, index) => ({ ...line, line_no: index + 1 })))
      message.success(`导入已确认并写入 ${result.pr_no}`)
      setImportOpen(false)
    } catch (error) {
      if (isApiUnavailable(error)) message.warning(friendlyUnavailable('确认导入', error))
      else message.error((error as Error).message)
    }
  }

  if (oaLoading) return <Spin />

  return (
    <div data-testid="erp-request-new-page">
      <PageTitle
        title="新建采购申请"
        subtitle={request ? `草稿 ${request.request_no}` : '支持人工录入、Excel 导入与 Agent 填报'}
        extra={<Button onClick={() => navigate('/erp/workbench')}>返回工作台</Button>}
      />
      {importHint && <Alert type="warning" showIcon message={importHint} data-testid="import-unavailable" />}
      <Card className="section-card" title="来源 OA（可选）">
        <Select
          showSearch
          allowClear
          style={{ width: 420 }}
          placeholder="选择已审批 OA 以承接建单"
          value={oa?.id}
          optionFilterProp="label"
          options={oaOptions.map((item) => ({
            value: item.id,
            label: `${item.application_no} · ${item.title} · ${getOAStatusMeta(item.status).label}`,
          }))}
          onChange={(id) => {
            const hit = oaOptions.find((item) => item.id === id)
            setOA(hit)
            if (hit) {
              form.setFieldsValue({
                department: hit.department,
                applicant: hit.applicant,
                purchase_reason: hit.title,
              })
              navigate(`/erp/requests/new?oa_id=${hit.id}`, { replace: true })
            }
          }}
          data-testid="erp-oa-source-select"
        />
      </Card>
      <Form form={form} layout="vertical" data-testid="erp-request-form" initialValues={{ purchase_type: 'inquiry', receive_address: '总部收货仓' }}>
        <Card title="单据头" className="section-card">
          <Row gutter={16}>
            <Col span={8}><Form.Item label="申请单号"><Input disabled value={request?.request_no || '保存草稿后生成'} data-testid="header-request-no" /></Form.Item></Col>
            <Col span={8}><Form.Item name="department" label="申请部门" rules={[{ required: true, message: '请填写申请部门' }]}><Input data-testid="header-department" /></Form.Item></Col>
            <Col span={8}><Form.Item name="applicant" label="申请人" rules={[{ required: true, message: '请填写申请人' }]}><Input data-testid="header-applicant" /></Form.Item></Col>
            <Col span={8}><Form.Item name="budget_project" label="预算项目" rules={[{ required: true, message: '请填写预算项目' }]}><Input data-testid="header-budget-project" /></Form.Item></Col>
            <Col span={8}><Form.Item name="cost_center" label="成本中心" rules={[{ required: true, message: '请填写成本中心' }]}><Input data-testid="header-cost-center" /></Form.Item></Col>
            <Col span={8}><Form.Item name="purchase_type" label="采购类型" rules={[{ required: true }]}><Select options={purchaseTypeOptions} data-testid="header-purchase-type" /></Form.Item></Col>
            <Col span={8}><Form.Item name="expected_delivery_date" label="期望到货日" rules={[{ required: true, message: '请选择期望到货日' }]}><DatePicker style={{ width: '100%' }} data-testid="header-expected-delivery-date" /></Form.Item></Col>
            <Col span={8}><Form.Item name="receive_address" label="收货地址" rules={[{ required: true }]}><Input data-testid="header-receive-address" /></Form.Item></Col>
            <Col span={24}><Form.Item name="purchase_reason" label="采购原因" rules={[{ required: true, message: '请填写采购原因' }, { max: 500 }]}><Input.TextArea rows={3} data-testid="header-purchase-reason" /></Form.Item></Col>
          </Row>
        </Card>
        <Card
          title="物资明细"
          className="section-card"
          extra={
            <Space>
              <Upload accept=".xlsx" showUploadList={false} beforeUpload={(file) => { void onImportUpload(file); return false }}>
                <Button icon={<CloudUpload size={16} strokeWidth={1.75} />} data-testid="excel-import-button">从 Excel 导入</Button>
              </Upload>
              <Button icon={<Plus size={16} strokeWidth={1.75} />} onClick={addLine} data-testid="add-material-line">添加物品</Button>
            </Space>
          }
        >
          <div className="line-grid line-grid-header"><span>行号</span><span>ERP 物料</span><span>规格</span><span>单位</span><span>数量</span><span>单价（元）</span><span>金额（元）</span><span>操作</span></div>
          {lines.map((line) => (
            <div className="line-grid" key={line.line_no} data-testid={`material-line-${line.line_no}`}>
              <strong>{line.line_no}</strong>
              <Select
                showSearch
                filterOption={false}
                onSearch={searchMaterials}
                value={line.material_code}
                placeholder={line.material_name ? `待匹配：${line.material_name}` : '输入编码/名称检索'}
                data-testid={`material-select-${line.line_no}`}
                options={materials.map((m) => ({ value: m.material_code, label: `${m.material_code} · ${m.material_name}`, material: m }))}
                onSelect={(_, option) => {
                  const material = option.material as Material
                  updateLine(line.line_no, {
                    material_code: material.material_code,
                    material_name: material.material_name,
                    specification: material.specification,
                    unit: material.unit,
                    unit_price: material.standard_price,
                  })
                }}
              />
              <Input value={line.specification} onChange={(e) => updateLine(line.line_no, { specification: e.target.value })} />
              <Input value={line.unit} onChange={(e) => updateLine(line.line_no, { unit: e.target.value })} />
              <InputNumber min={0.0001} value={line.quantity as number | undefined} onChange={(v) => updateLine(line.line_no, { quantity: v || undefined })} />
              <InputNumber min={0} precision={2} value={line.unit_price as number | undefined} onChange={(v) => updateLine(line.line_no, { unit_price: v ?? undefined })} />
              <strong>¥{lineAmount(line).toFixed(2)}</strong>
              <Popconfirm title="删除该物资行？" onConfirm={() => removeLine(line.line_no)}>
                <Button danger type="text" icon={<Trash2 size={16} strokeWidth={1.75} />} disabled={lines.length === 1} />
              </Popconfirm>
            </div>
          ))}
          <Divider />
          <div className="amount-summary" data-testid="total-amount"><Statistic title="采购总金额" value={amount} precision={2} prefix="¥" /></div>
        </Card>
        <Card title="附件" className="section-card">
          <Upload.Dragger fileList={files} onChange={({ fileList }) => setFiles(fileList)} beforeUpload={() => false} multiple>
            <CloudUpload className="upload-icon" size={28} strokeWidth={1.5} />
            <p>附件仅生成 demo:// 引用</p>
          </Upload.Dragger>
        </Card>
        <div className="form-actions">
          <Button onClick={() => navigate('/erp/workbench')}>取消</Button>
          <Button onClick={saveDraft} data-testid="save-draft-button">保存草稿</Button>
          <Button type="primary" danger loading={submitting} onClick={confirmSubmit} data-testid="submit-procurement-button">提交审批</Button>
        </div>
      </Form>

      <Modal
        open={importOpen}
        title="Excel 导入预览"
        width={900}
        onCancel={() => setImportOpen(false)}
        onOk={confirmImport}
        okText="确认导入"
        okButtonProps={{ 'data-testid': 'import-confirm-ok' } as React.ComponentProps<typeof Button>}
      >
        {importBatch ? (
          <>
            <Typography.Paragraph>
              文件 {importBatch.filename}：共 {importBatch.total_rows} 行，成功 {importBatch.success_rows}，失败 {importBatch.failed_rows}
            </Typography.Paragraph>
            <Table
              size="small"
              rowKey="row_no"
              dataSource={importBatch.rows || []}
              data-testid="import-preview-table"
              pagination={false}
              columns={[
                { title: '行', dataIndex: 'row_no' },
                { title: '原始名称', dataIndex: 'raw_material_name' },
                { title: '匹配编码', dataIndex: 'material_code' },
                { title: '数量', dataIndex: 'quantity' },
                { title: '状态', dataIndex: 'status' },
                { title: '说明', dataIndex: 'message' },
              ]}
            />
          </>
        ) : <Spin />}
      </Modal>
    </div>
  )
}
