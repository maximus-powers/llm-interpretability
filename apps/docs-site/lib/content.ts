import fs from 'fs'
import path from 'path'
import matter from 'gray-matter'

const contentDirectory = path.join(process.cwd(), 'content')

export interface Article {
  slug: string
  title: string
  section: string
  category?: string
  order?: number
}

export interface Section {
  name: string
  slug: string
  categories: Category[]
}

export interface Category {
  name: string
  slug: string
  articles: Article[]
}

function getArticlesFromDir(dir: string, section: string, category?: string): Article[] {
  if (!fs.existsSync(dir)) return []
  
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.mdx'))
  
  return files.map(filename => {
    const filePath = path.join(dir, filename)
    const fileContents = fs.readFileSync(filePath, 'utf8')
    const { data } = matter(fileContents)
    
    return {
      slug: filename.replace('.mdx', ''),
      title: data.title || filename.replace('.mdx', '').replace(/-/g, ' '),
      section,
      category,
      order: data.order || 999,
    }
  }).sort((a, b) => a.order - b.order)
}

export function getNavigation(): Section[] {
  const sections: Section[] = []
  
  // Research section
  const researchDir = path.join(contentDirectory, 'research')
  if (fs.existsSync(researchDir)) {
    const categories: Category[] = []
    const categoryDirs = fs.readdirSync(researchDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
    
    const categoryOrder = ['foundations', 'interpretability', 'representation-engineering', 'weight-space', 'advanced']
    const categoryNames: Record<string, string> = {
      'foundations': 'Foundations',
      'interpretability': 'Interpretability',
      'representation-engineering': 'Representation Engineering',
      'weight-space': 'Weight Space',
      'advanced': 'Advanced',
    }
    
    categoryDirs.sort((a, b) => {
      const aIdx = categoryOrder.indexOf(a.name)
      const bIdx = categoryOrder.indexOf(b.name)
      return (aIdx === -1 ? 999 : aIdx) - (bIdx === -1 ? 999 : bIdx)
    })
    
    for (const dir of categoryDirs) {
      const articles = getArticlesFromDir(
        path.join(researchDir, dir.name),
        'research',
        dir.name
      )
      if (articles.length > 0) {
        categories.push({
          name: categoryNames[dir.name] || dir.name,
          slug: dir.name,
          articles,
        })
      }
    }
    
    sections.push({
      name: 'Research',
      slug: 'research',
      categories,
    })
  }
  
  // Project section
  const projectDir = path.join(contentDirectory, 'project')
  if (fs.existsSync(projectDir)) {
    const articles = getArticlesFromDir(projectDir, 'project')
    if (articles.length > 0) {
      sections.push({
        name: 'Project',
        slug: 'project',
        categories: [{
          name: 'Guides',
          slug: '',
          articles,
        }],
      })
    }
  }
  
  return sections
}

export function getArticleContent(section: string, category: string | null, slug: string): string | null {
  let filePath: string
  
  if (category) {
    filePath = path.join(contentDirectory, section, category, `${slug}.mdx`)
  } else {
    filePath = path.join(contentDirectory, section, `${slug}.mdx`)
  }
  
  if (!fs.existsSync(filePath)) return null
  
  return fs.readFileSync(filePath, 'utf8')
}

export function getAllArticlePaths(): { section: string; category?: string; slug: string }[] {
  const paths: { section: string; category?: string; slug: string }[] = []
  
  // Research articles
  const researchDir = path.join(contentDirectory, 'research')
  if (fs.existsSync(researchDir)) {
    const categoryDirs = fs.readdirSync(researchDir, { withFileTypes: true })
      .filter(d => d.isDirectory())
    
    for (const dir of categoryDirs) {
      const files = fs.readdirSync(path.join(researchDir, dir.name))
        .filter(f => f.endsWith('.mdx'))
      
      for (const file of files) {
        paths.push({
          section: 'research',
          category: dir.name,
          slug: file.replace('.mdx', ''),
        })
      }
    }
  }
  
  // Project articles
  const projectDir = path.join(contentDirectory, 'project')
  if (fs.existsSync(projectDir)) {
    const files = fs.readdirSync(projectDir)
      .filter(f => f.endsWith('.mdx'))
    
    for (const file of files) {
      paths.push({
        section: 'project',
        slug: file.replace('.mdx', ''),
      })
    }
  }
  
  return paths
}

export function getResearchStaticParams() {
  const paths = getAllArticlePaths()
  return paths
    .filter(p => p.section === 'research' && p.category)
    .map(p => ({
      category: p.category,
      slug: p.slug,
    }))
}

export function getProjectStaticParams() {
  const paths = getAllArticlePaths()
  return paths
    .filter(p => p.section === 'project')
    .map(p => ({
      slug: p.slug,
    }))
}
