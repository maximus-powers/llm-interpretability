'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Section } from '../lib/content'

export function Nav({ sections }: { sections: Section[] }) {
  const pathname = usePathname()
  
  return (
    <nav className="fixed top-0 left-0 w-70 h-screen bg-[#0a0a0a] border-r border-[#222] py-8 overflow-y-auto">
      <div className="px-6 mb-8">
        <Link href="/" className="text-[#e5e5e5] font-semibold text-[0.95rem] hover:text-[#60a5fa]">
          Weight-Space Learning
        </Link>
      </div>
      
      {sections.map(section => (
        <div key={section.slug}>
          {section.categories.map(category => (
            <div key={category.slug} className="mb-6">
              <div className="text-[0.7rem] uppercase tracking-wide text-[#555] px-6 mb-2">
                {section.name}{category.name !== 'Guides' ? ` / ${category.name}` : ''}
              </div>
              <ul className="list-none">
                {category.articles.map(article => {
                  const href = category.slug 
                    ? `/${section.slug}/${category.slug}/${article.slug}`
                    : `/${section.slug}/${article.slug}`
                  const isActive = pathname === href
                  
                  return (
                    <li key={article.slug}>
                      <Link 
                        href={href} 
                        className={`block py-1.5 px-6 text-[0.85rem] transition-colors ${
                          isActive 
                            ? 'text-[#3b82f6] bg-[#111]' 
                            : 'text-[#888] hover:text-[#e5e5e5] hover:bg-[#111]'
                        }`}
                      >
                        {article.title}
                      </Link>
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </div>
      ))}
    </nav>
  )
}
