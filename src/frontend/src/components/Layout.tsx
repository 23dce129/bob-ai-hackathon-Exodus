import { NavLink, Outlet } from 'react-router-dom'
import { DisclaimerBanner } from './DisclaimerBanner'

const navItems = [
  { to: '/',            label: 'Dashboard',          icon: '⬛' },
  { to: '/signals',     label: 'Signal Detection',    icon: '🔍' },
  { to: '/readiness',   label: 'Submission Readiness',icon: '📋' },
  { to: '/gap-report',  label: 'Gap Report',          icon: '🗂' },
  { to: '/action-brief',label: 'Action Brief',        icon: '⚡' },
]

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Top header */}
      <header className="bg-pharma-800 text-white flex items-center gap-3 px-6 py-3 shadow-md z-10">
        <div className="flex items-center gap-2">
          <span className="text-xl font-bold tracking-tight">PharmaGuard AI</span>
          <span className="text-pharma-300 text-xs font-normal ml-1">
            Pharmacovigilance · Regulatory Readiness
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="inline-flex items-center bg-pharma-700 rounded px-2 py-0.5 text-xs text-pharma-200">
            Synthetic Demo Dataset
          </span>
        </div>
      </header>

      <DisclaimerBanner />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <nav className="w-56 bg-white border-r border-gray-200 flex flex-col py-4 gap-1 shrink-0">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-4 mb-1">
            Navigation
          </p>
          {navItems.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2 text-sm rounded-md mx-2 transition-colors ${
                  isActive
                    ? 'bg-pharma-50 text-pharma-800 font-medium border-l-2 border-pharma-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`
              }
            >
              <span className="text-base leading-none">{icon}</span>
              {label}
            </NavLink>
          ))}

          <div className="mt-auto px-4 py-3 border-t border-gray-100 mx-2">
            <p className="text-xs text-gray-400">ICH M4 CTD · FDA/EMA</p>
            <p className="text-xs text-gray-400">FAERS Sample Dataset</p>
          </div>
        </nav>

        {/* Page content */}
        <main className="flex-1 overflow-auto bg-gray-50 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
