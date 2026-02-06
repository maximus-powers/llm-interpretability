import type { Metadata } from 'next'
import { Nav } from '../components/Nav'
import { getNavigation } from '../lib/content'
import './globals.css'

export const metadata: Metadata = {
  title: 'Weight-Space Learning',
  description: 'Research notes on representation engineering and weight-space learning',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const sections = getNavigation()
  
  return (
    <html lang="en">
      <body className="font-mono bg-[#0a0a0a] text-[#e5e5e5] leading-relaxed antialiased">
        <Nav sections={sections} />
        <main className="ml-70 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}
