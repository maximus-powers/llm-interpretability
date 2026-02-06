import { notFound } from 'next/navigation'
import { getArticleContent, getResearchStaticParams } from '../../../../lib/content'
import matter from 'gray-matter'
import { MDXRemote } from '../../../../components/MDXRemote'

export const generateStaticParams = getResearchStaticParams

export default async function ResearchArticle({ params }: {
  params: Promise<{ category: string; slug: string }>
}) {
  const { category, slug } = await params
  const content = getArticleContent('research', category, slug)
  
  if (!content) {
    notFound()
  }
  
  const { content: mdxContent } = matter(content)
  
  return (
    <article className="max-w-2xl mx-auto px-6 py-12 prose prose-invert prose-sm">
      <MDXRemote source={mdxContent} />
    </article>
  )
}
