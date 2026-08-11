import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Input,
  List,
  Popconfirm,
  Select,
  Space,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd'
import {
  CloseOutlined,
  MinusOutlined,
  RobotOutlined,
  DragOutlined,
  ExpandOutlined,
  SendOutlined,
  PauseCircleOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { useLocation, useNavigate } from 'react-router-dom'
import { api, friendlyUnavailable, isApiUnavailable } from '../api'
import { currentRunnableStep, executeAgentStep, type AgentStep } from '../agent/domDriver'
import { pickExcelDirectory } from '../agent/folderPicker'
import type { TaskStatus } from '../types'

type WindowMode = 'collapsed' | 'expanded' | 'minimized' | 'monitor'

type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }

type AgentTaskView = TaskStatus & {
  steps?: AgentStep[]
  messages?: ChatMessage[]
  waiting?: {
    type?: string
    files?: Array<{ name: string; path: string }>
    options?: Array<{ id: number; application_no: string; title: string; department?: string }>
    errors?: string[]
    department?: string
  }
  chips?: Array<{ id: string; label: string }>
}

const DEFAULT_CHIPS = [
  { id: 'import_purchase_to_oa', label: '帮我导入生产部的采购申请到 OA' },
  { id: 'submit_approved_purchase', label: '处理已通过的采购申请' },
  { id: 'view_current_task', label: '查看当前任务' },
  { id: 'resume_last_task', label: '继续上次任务' },
]

function resolveBusinessKey(pathname: string) {
  const oa = pathname.match(/^\/oa\/([^/]+)/)
  if (oa) return `OA#${oa[1]}`
  const pr = pathname.match(/^\/procurement\/(?!new$)([^/]+)/)
  if (pr) return pr[1]
  const po = pathname.match(/^\/erp\/orders\/([^/]+)/)
  if (po) return po[1]
  if (pathname.includes('/erp/workbench')) return 'workbench'
  if (pathname.includes('/erp/export')) return 'export-batch'
  if (pathname.includes('/erp/requests/new') || pathname.includes('/procurement/new')) return 'pr-draft'
  return pathname
}

function statusColor(status?: string) {
  if (status === 'completed') return 'success'
  if (status === 'wait_user') return 'warning'
  if (status === 'failed' || status === 'stopped') return 'error'
  if (status === 'running') return 'processing'
  return 'default'
}

export function AgentFloatingWindow() {
  const location = useLocation()
  const navigate = useNavigate()
  const [mode, setMode] = useState<WindowMode>('collapsed')
  const [input, setInput] = useState('')
  const [task, setTask] = useState<AgentTaskView>()
  const [online, setOnline] = useState(true)
  const [busy, setBusy] = useState(false)
  const [selectedExcelPath, setSelectedExcelPath] = useState<string>()
  const [selectedOaId, setSelectedOaId] = useState<number>()
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const dragRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null)
  const runningRef = useRef(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const businessKey = resolveBusinessKey(location.pathname)
  const chips = task?.chips?.length ? task.chips : DEFAULT_CHIPS
  const steps = task?.steps || []
  const chatMessages = useMemo(() => task?.messages || [], [task?.messages])
  const currentStep = useMemo(() => currentRunnableStep(steps) || steps.find((step) => step.status === 'running'), [steps])
  const lastAssistant = useMemo(
    () => [...chatMessages].reverse().find((item) => item.role === 'assistant')?.content,
    [chatMessages],
  )

  const refreshTask = async (taskId: string) => {
    const next = await api.getAgentTask(taskId) as AgentTaskView
    setTask(next)
    return next
  }

  useEffect(() => {
    api.getAgentActive()
      .then((data) => {
        setOnline(true)
        if (data?.task_id) void refreshTask(data.task_id).catch(() => undefined)
      })
      .catch((error) => {
        setOnline(!isApiUnavailable(error))
      })
  }, [location.pathname])

  // While DOM steps run, stay in a thin monitor strip so the form (e.g. 存为草稿) stays clickable / screenshot-clean.
  useEffect(() => {
    if (task?.status === 'running' && !task.is_paused) {
      setMode((prev) => (prev === 'collapsed' ? prev : 'monitor'))
      return
    }
    if (task?.status === 'wait_user') {
      setMode((prev) => (prev === 'monitor' ? 'expanded' : prev))
    }
  }, [task?.status, task?.is_paused])

  useEffect(() => {
    if (!task?.task_id) return
    if (task.status !== 'running' || task.is_paused) return
    if (runningRef.current) return
    const step = currentRunnableStep(steps)
    if (!step) return
    runningRef.current = true
    ;(async () => {
      try {
        const result = await executeAgentStep(step, navigate)
        const next = await api.reportAgentStepResult(task.task_id, {
          step_id: step.step_id,
          status: result.ok ? 'passed' : 'failed',
          actual: result.actual,
          detail: result.actual,
        })
        setTask(next as AgentTaskView)
      } catch (error) {
        try {
          const next = await api.reportAgentStepResult(task.task_id, {
            step_id: step.step_id,
            status: 'failed',
            actual: { error: (error as Error).message },
          })
          setTask(next as AgentTaskView)
        } catch (reportError) {
          message.error((reportError as Error).message)
        }
      } finally {
        runningRef.current = false
      }
    })()
  }, [task?.task_id, task?.status, task?.is_paused, steps, navigate])

  const onDragStart = (event: React.MouseEvent) => {
    event.preventDefault()
    dragRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      originX: offset.x,
      originY: offset.y,
    }
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return
      setOffset({
        x: dragRef.current.originX + (ev.clientX - dragRef.current.startX),
        y: dragRef.current.originY + (ev.clientY - dragRef.current.startY),
      })
    }
    const onUp = () => {
      dragRef.current = null
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  const sendMessage = async (text: string) => {
    const content = text.trim()
    if (!content) return
    try {
      setBusy(true)
      const data = await api.agentChat({
        message: content,
        route: location.pathname,
        business_key: businessKey,
      })
      if (data.task) setTask(data.task as AgentTaskView)
      else if (data.reply) message.info(data.reply)
      setInput('')
    } catch (error) {
      if (isApiUnavailable(error)) message.warning(friendlyUnavailable('Agent chat', error))
      else message.error((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const pickFolder = async () => {
    if (!task?.task_id) return
    try {
      setBusy(true)
      const picked = await pickExcelDirectory()
      if (!picked) {
        message.warning('当前浏览器不支持系统文件夹选择，请改用下方上传 Excel 文件')
        fileInputRef.current?.click()
        return
      }
      if (!picked.files.length) {
        message.warning(`文件夹「${picked.folderName}」中没有 Excel`)
        return
      }
      if (picked.files.length === 1) {
        const next = await api.continueAgentTask(task.task_id, { file: picked.files[0].file })
        setTask(next as AgentTaskView)
        return
      }
      // Multiple files: upload is still needed; present local names via waiting by uploading none and showing picker UI.
      message.info(`文件夹内有 ${picked.files.length} 个 Excel，请在下方选择并上传其中一个`)
      // Stash on window for quick select UI via Upload list — user picks one file manually.
      ;(window as unknown as { __agentPickedExcels?: typeof picked.files }).__agentPickedExcels = picked.files
    } catch (error) {
      if ((error as Error).name === 'AbortError') return
      message.error((error as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const continueWithUpload = async (file: File) => {
    if (!task?.task_id) return
    const next = await api.continueAgentTask(task.task_id, { file })
    setTask(next as AgentTaskView)
  }

  const continueSelectExcel = async () => {
    if (!task?.task_id || !selectedExcelPath) return message.warning('请选择 Excel')
    const next = await api.continueAgentTask(task.task_id, { excel_path: selectedExcelPath })
    setTask(next as AgentTaskView)
  }

  const continueSelectOa = async () => {
    if (!task?.task_id || selectedOaId == null) return message.warning('请选择 OA 申请')
    const next = await api.continueAgentTask(task.task_id, { oa_id: selectedOaId })
    setTask(next as AgentTaskView)
  }

  const pause = async () => {
    if (!task?.task_id) return message.warning('暂无任务可暂停')
    setTask(await api.pauseAgentTask(task.task_id) as AgentTaskView)
    message.success('已暂停')
  }

  const stop = async () => {
    if (!task?.task_id) return message.warning('暂无任务可停止')
    setTask(await api.stopAgentTask(task.task_id) as AgentTaskView)
    message.success('已紧急停止')
  }

  const resume = async () => {
    if (!task?.task_id) return
    setTask(await api.resumeAgentTask(task.task_id) as AgentTaskView)
  }

  const reset = async () => {
    await api.resetDemo()
    setTask(undefined)
    message.success('演示数据已重置')
  }

  const style = { transform: `translate(${offset.x}px, ${offset.y}px)` }
  const waiting = task?.waiting

  if (mode === 'collapsed') {
    return (
      <button
        type="button"
        className="agent-fab"
        style={style}
        data-testid="agent-floating-window"
        aria-label="打开 Agent 浮窗"
        onClick={() => setMode('expanded')}
      >
        <RobotOutlined />
        <span className="agent-fab-dot" />
      </button>
    )
  }

  if (mode === 'minimized') {
    return (
      <div className="agent-float agent-float-mini" style={style} data-testid="agent-floating-window">
        <button type="button" className="agent-drag-handle" onMouseDown={onDragStart} data-testid="agent-drag-handle">
          <DragOutlined />
        </button>
        <Typography.Text>{task?.status || 'idle'} · {task?.task_id || '无任务'}</Typography.Text>
        <Button type="text" size="small" onClick={() => setMode('monitor')} data-testid="agent-monitor-button">监控</Button>
        <Button type="text" size="small" icon={<ExpandOutlined />} onClick={() => setMode('expanded')} data-testid="agent-expand-button" />
        <Button type="text" size="small" icon={<CloseOutlined />} onClick={() => setMode('collapsed')} data-testid="agent-collapse-button" />
      </div>
    )
  }

  if (mode === 'monitor') {
    return (
      <div className="agent-float agent-float-monitor" style={style} data-testid="agent-floating-window">
        <button type="button" className="agent-drag-handle" onMouseDown={onDragStart} data-testid="agent-drag-handle" aria-label="拖动监控条">
          <DragOutlined />
        </button>
        <RobotOutlined />
        <Tag color={online ? 'success' : 'default'} data-testid="agent-online-tag">{online ? '在线' : '离线'}</Tag>
        {task?.status && <Tag color={statusColor(task.status)} data-testid="agent-task-status">{task.status}</Tag>}
        <Typography.Text className="agent-monitor-step" ellipsis data-testid="agent-monitor-step">
          {currentStep ? `${currentStep.step_id} · ${currentStep.title || ''}` : (lastAssistant || '待命')}
        </Typography.Text>
        <Space size={4}>
          <Button type="text" size="small" icon={<ExpandOutlined />} onClick={() => setMode('expanded')} data-testid="agent-expand-button">详情</Button>
          <Button type="text" size="small" icon={<MinusOutlined />} onClick={() => setMode('minimized')} data-testid="agent-minimize-button" />
          <Button type="text" size="small" icon={<CloseOutlined />} onClick={() => setMode('collapsed')} data-testid="agent-close-button" />
        </Space>
      </div>
    )
  }

  return (
    <div className="agent-float agent-float-panel agent-chat-panel" style={style} data-testid="agent-floating-window">
      <div className="agent-float-header" onMouseDown={onDragStart} data-testid="agent-drag-handle">
        <Space>
          <DragOutlined />
          <RobotOutlined />
          <strong>GUI Agent</strong>
          <Tag color={online ? 'success' : 'default'} data-testid="agent-online-tag">{online ? '在线' : '离线'}</Tag>
          {task?.status && <Tag color={statusColor(task.status)} data-testid="agent-task-status">{task.status}</Tag>}
        </Space>
        <Space>
          <Button type="text" size="small" onClick={() => setMode('monitor')} data-testid="agent-monitor-button">监控条</Button>
          <Button type="text" size="small" icon={<MinusOutlined />} onClick={() => setMode('minimized')} data-testid="agent-minimize-button" />
          <Button type="text" size="small" icon={<CloseOutlined />} onClick={() => setMode('collapsed')} data-testid="agent-close-button" />
        </Space>
      </div>
      <div className="agent-float-body">
        <div className="agent-chip-row" data-testid="agent-quick-chips">
          {chips.map((chip) => (
            <Button key={chip.id} size="small" onClick={() => void sendMessage(chip.label)} data-testid={`agent-chip-${chip.id}`}>
              {chip.label}
            </Button>
          ))}
        </div>

        <div className="agent-chat-log" data-testid="agent-chat-log">
          {chatMessages.length === 0 && (
            <Typography.Paragraph type="secondary">输入自然语言任务，或点击上方快捷指令。</Typography.Paragraph>
          )}
          {chatMessages.map((item, index) => (
            <div key={`${item.role}-${index}`} className={`agent-chat-bubble agent-chat-${item.role}`}>
              <Typography.Text strong>{item.role === 'user' ? '你' : 'Agent'}</Typography.Text>
              <div>{item.content}</div>
            </div>
          ))}
        </div>

        {waiting?.type === 'select_folder' && (
          <Alert
            className="section-card"
            type="warning"
            showIcon
            data-testid="agent-wait-folder"
            message="需要选择文件夹"
            description={(
              <Space direction="vertical">
                <span>未提供路径时不会猜测目录，请选择包含采购申请 Excel 的文件夹，或直接上传 xlsx。</span>
                <Space wrap>
                  <Button type="primary" onClick={() => void pickFolder()} data-testid="agent-pick-folder-button">选择文件夹</Button>
                  <Upload
                    accept=".xlsx,.xlsm"
                    showUploadList={false}
                    beforeUpload={(file) => {
                      void continueWithUpload(file)
                      return false
                    }}
                  >
                    <Button data-testid="agent-upload-excel-button">上传 Excel</Button>
                  </Upload>
                </Space>
              </Space>
            )}
          />
        )}

        {waiting?.type === 'select_excel' && (
          <Alert
            className="section-card"
            type="warning"
            showIcon
            data-testid="agent-wait-excel"
            message="请确认 Excel 文件"
            description={(
              <Space direction="vertical" style={{ width: '100%' }}>
                <Select
                  style={{ width: '100%' }}
                  placeholder="选择文件"
                  value={selectedExcelPath}
                  onChange={setSelectedExcelPath}
                  options={(waiting.files || []).map((file) => ({ value: file.path, label: file.name }))}
                  data-testid="agent-excel-select"
                />
                <Button type="primary" onClick={() => void continueSelectExcel()} data-testid="agent-confirm-excel-button">确认文件</Button>
              </Space>
            )}
          />
        )}

        {waiting?.type === 'select_oa' && (
          <Alert
            className="section-card"
            type="warning"
            showIcon
            data-testid="agent-wait-oa"
            message="请选择目标 OA 申请"
            description={(
              <Space direction="vertical" style={{ width: '100%' }}>
                <Select
                  style={{ width: '100%' }}
                  placeholder="多条命中，请选择"
                  value={selectedOaId}
                  onChange={setSelectedOaId}
                  options={(waiting.options || []).map((item) => ({
                    value: item.id,
                    label: `${item.application_no} · ${item.title}`,
                  }))}
                  data-testid="agent-oa-select"
                />
                <Button type="primary" onClick={() => void continueSelectOa()} data-testid="agent-confirm-oa-button">确认并继续</Button>
              </Space>
            )}
          />
        )}

        {waiting?.type === 'validation_errors' && (
          <Alert
            className="section-card"
            type="error"
            showIcon
            data-testid="agent-wait-validation"
            message="Excel 校验失败"
            description={(
              <Space direction="vertical">
                <ul>{(waiting.errors || []).map((item) => <li key={item}>{item}</li>)}</ul>
                <Upload accept=".xlsx,.xlsm" showUploadList={false} beforeUpload={(file) => { void continueWithUpload(file); return false }}>
                  <Button data-testid="agent-reupload-excel-button">重新上传 Excel</Button>
                </Upload>
              </Space>
            )}
          />
        )}

        {waiting?.type === 'step_failed' && (
          <Alert
            className="section-card"
            type="error"
            showIcon
            data-testid="agent-wait-step-failed"
            message="步骤失败，等待处理"
            description={(
              <Button onClick={() => task?.task_id && void api.continueAgentTask(task.task_id, {}).then((next) => setTask(next as AgentTaskView))} data-testid="agent-retry-step-button">
                重试当前步骤
              </Button>
            )}
          />
        )}

        <div className="agent-step-list" data-testid="agent-step-list">
          <Typography.Text strong>执行步骤</Typography.Text>
          <List
            size="small"
            dataSource={steps}
            locale={{ emptyText: '暂无步骤' }}
            renderItem={(step) => (
              <List.Item data-testid={`agent-step-${step.step_id}`}>
                <Space direction="vertical" size={0} style={{ width: '100%' }}>
                  <Space>
                    <Tag color={statusColor(step.status)}>{step.status || 'pending'}</Tag>
                    <span>{step.title || step.step_id}</span>
                    <Typography.Text type="secondary">retry={step.retry_count || 0}</Typography.Text>
                  </Space>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    expected: {step.expected || '-'}
                  </Typography.Text>
                </Space>
              </List.Item>
            )}
          />
        </div>

        <Space.Compact block className="agent-chat-input">
          <Input
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onPressEnter={() => void sendMessage(input)}
            placeholder="描述你希望 Agent 完成的任务…"
            data-testid="agent-chat-input"
          />
          <Button type="primary" icon={<SendOutlined />} loading={busy} onClick={() => void sendMessage(input)} data-testid="agent-send-button">
            发送
          </Button>
        </Space.Compact>

        <Space wrap className="agent-float-actions">
          <Button icon={<PauseCircleOutlined />} onClick={() => void pause()} data-testid="agent-pause-button">暂停</Button>
          {task?.status === 'paused' && <Button onClick={() => void resume()} data-testid="agent-resume-button">继续</Button>}
          <Button danger icon={<StopOutlined />} onClick={() => void stop()} data-testid="agent-stop-button">停止</Button>
          <Popconfirm title="确认重置演示数据？" onConfirm={() => void reset()}>
            <Button danger data-testid="reset-demo-button">重置演示</Button>
          </Popconfirm>
        </Space>
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xlsm"
          hidden
          data-testid="agent-hidden-file-input"
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) void continueWithUpload(file)
            event.currentTarget.value = ''
          }}
        />
      </div>
    </div>
  )
}
