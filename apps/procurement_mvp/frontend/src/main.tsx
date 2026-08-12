import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import './styles.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#164b8f',
          borderRadius: 4,
          colorBgLayout: '#f5f7fa',
          colorBorder: '#d9e2ec',
          colorText: '#1f2937',
          colorTextSecondary: '#64748b',
          fontFamily: '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
          controlHeight: 32,
        },
        components: {
          Layout: { headerBg: '#ffffff', siderBg: '#ffffff', headerHeight: 56 },
          Card: { borderRadiusLG: 4 },
          Button: { borderRadius: 4 },
          Tag: { borderRadiusSM: 2 },
        },
      }}
    >
      <BrowserRouter><App /></BrowserRouter>
    </ConfigProvider>
  </React.StrictMode>,
)
