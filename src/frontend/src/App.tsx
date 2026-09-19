import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Dashboard } from './pages/Dashboard'
import { SignalDetection } from './pages/SignalDetection'
import { SignalDetails } from './pages/SignalDetails'
import { SubmissionReadiness } from './pages/SubmissionReadiness'
import { GapReport } from './pages/GapReport'
import { ActionBrief } from './pages/ActionBrief'

export default function App() {
  return (
    <ErrorBoundary fallbackTitle="Application Render Error">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="signals" element={<SignalDetection />} />
            <Route path="signals/:drug/:event" element={<SignalDetails />} />
            <Route path="readiness" element={<SubmissionReadiness />} />
            <Route path="gap-report" element={<GapReport />} />
            <Route path="action-brief" element={<ActionBrief />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}
