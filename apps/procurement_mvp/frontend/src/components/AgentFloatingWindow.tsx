import { useEffect, useRef, useState } from 'react'
import { Alert, Button, Descriptions, Input, Popconfirm, Space, Tag, Typography, message } from 'antd'
import {
  CloseOutlined,
  MinusOutlined,
  RobotOutlined,
  DragOutlined,
  ExpandOutlined,
} from '@ant-design/icons'
import { useLocation } from 'react-router-dom'
import { api, friendlyUnavailable, isApiUnavailable } from '../api'
import type { AgentActiveTask, TaskStatus } from '../types'

type WindowMode = 'collapsed' | 'expanded' | 'minimized'

let taskSequence = 0
const nextTaskId = (operation: string) => `web-${operation}-${Date.now()}-${++taskSequence}`

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

export function AgentFloatingWindow() {
  const location = useLocation()
  const [mode, setMode] = useState<WindowMode>('collapsed')
  const [taskId, setTaskId] = useState('')
  const [task, setTask] = useState<TaskStatus>()
  const [active, setActive] = useState<AgentActiveTask | null>(null)
  const [apiHint, setApiHint] = useState('')
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const dragRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null)

  const businessKey = resolveBusinessKey(location.pathname)

  useEffect(() => {
    api.getAgentActive()
      .then((data) => setActive(data))
      .catch((error) => {
        if (isApiUnavailable(error)) setApiHint(friendlyUnavailable('Agent active', error))
        else setApiHint('')
      })
  }, [location.pathname])

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

  const query = async () => {
    if (!taskId.trim()) return message.warning('请输入 task_id')
    try {
      setTask(await api.getTask(taskId.trim()))
      setApiHint('')
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  const reset = async () => {
    try {
      await api.resetDemo()
      setTask(undefined)
      setTaskId('')
      setActive(null)
      message.success('演示数据已重置')
    } catch (e) {
      message.error((e as Error).message)
    }
  }

  const pause = async () => {
    const id = task?.task_id || active?.task_id || taskId
    if (!id) return message.warning('暂无任务可暂停')
    try {
      const result = await api.pauseAgentTask(id)
      setTask(result)
      message.success('已请求暂停')
    } catch (e) {
      if (isApiUnavailable(e)) message.warning(friendlyUnavailable('暂停任务', e))
      else message.error((e as Error).message)
    }
  }

  const stop = async () => {
    const id = task?.task_id || active?.task_id || taskId
    if (!id) return message.warning('暂无任务可停止')
    try {
      const result = await api.stopAgentTask(id)
      setTask(result)
      message.success('已请求紧急停止')
    } catch (e) {
      if (isApiUnavailable(e)) message.warning(friendlyUnavailable('停止任务', e))
      else message.error((e as Error).message)
    }
  }

  const style = {
    transform: `translate(${offset.x}px, ${offset.y}px)`,
  }

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
        <Typography.Text>{task?.status || active?.status || 'idle'} · {task?.task_id || active?.task_id || '无任务'}</Typography.Text>
        <Button type="text" size="small" icon={<ExpandOutlined />} onClick={() => setMode('expanded')} data-testid="agent-expand-button" />
        <Button type="text" size="small" icon={<CloseOutlined />} onClick={() => setMode('collapsed')} data-testid="agent-collapse-button" />
      </div>
    )
  }

  return (
    <div className="agent-float agent-float-panel" style={style} data-testid="agent-floating-window">
      <div className="agent-float-header" onMouseDown={onDragStart} data-testid="agent-drag-handle">
        <Space>
          <DragOutlined />
          <RobotOutlined />
          <strong>GUI Agent</strong>
          <Tag color="success">在线</Tag>
        </Space>
        <Space>
          <Button type="text" size="small" icon={<MinusOutlined />} onClick={() => setMode('minimized')} data-testid="agent-minimize-button" />
          <Button type="text" size="small" icon={<CloseOutlined />} onClick={() => setMode('collapsed')} data-testid="agent-close-button" />
        </Space>
      </div>
      <div className="agent-float-body">
        <Alert
          type="info"
          showIcon
          message="页面上下文"
          description={`路由：${location.pathname}；业务键：${businessKey}`}
          data-testid="agent-context"
        />
        {apiHint && <Alert className="section-card" type="warning" showIcon message={apiHint} />}
        <div className="agent-float-section">
          <Typography.Text strong>任务查询</Typography.Text>
          <Space.Compact block style={{ marginTop: 8 }}>
            <Input
              value={taskId}
              onChange={(e) => setTaskId(e.target.value)}
              onPressEnter={query}
              placeholder="输入 task_id"
              data-testid="agent-task-id-input"
            />
            <Button type="primary" onClick={query} data-testid="agent-query-button">仅查询任务</Button>
          </Space.Compact>
        </div>
        {task && (
          <Descriptions bordered size="small" column={1} className="task-result" data-testid="agent-task-result">
            <Descriptions.Item label="task_id">{task.task_id}</Descriptions.Item>
            <Descriptions.Item label="business_key">{task.business_key}</Descriptions.Item>
            <Descriptions.Item label="操作">{task.operation}</Descriptions.Item>
            <Descriptions.Item label="状态"><Tag>{task.status}</Tag></Descriptions.Item>
          </Descriptions>
        )}
        <Space wrap className="agent-float-actions">
          <Button onClick={query} data-testid="agent-quick-query">仅查询任务</Button>
          <Button onClick={pause} data-testid="agent-pause-button">暂停</Button>
          <Button danger onClick={stop} data-testid="agent-stop-button">紧急停止</Button>
          <Popconfirm title="确认重置演示数据？" onConfirm={reset}>
            <Button danger data-testid="reset-demo-button">重置演示</Button>
          </Popconfirm>
        </Space>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 0, fontSize: 12 }}>
          关闭浮窗不会停止任务；task_id={nextTaskId('agent-ui')} 仅用于本地追踪。
        </Typography.Paragraph>
      </div>
    </div>
  )
}
