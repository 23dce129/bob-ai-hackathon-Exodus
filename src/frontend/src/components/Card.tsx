import React from 'react'

interface Props {
  children: React.ReactNode
  className?: string
}

export function Card({ children, className = '' }: Props) {
  return (
    <div className={`bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      {children}
    </div>
  )
}

export function CardHeader({ children, className = '' }: Props) {
  return (
    <div className={`px-5 py-4 border-b border-gray-100 ${className}`}>
      {children}
    </div>
  )
}

export function CardBody({ children, className = '' }: Props) {
  return (
    <div className={`px-5 py-4 ${className}`}>
      {children}
    </div>
  )
}
