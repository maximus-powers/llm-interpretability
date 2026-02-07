import { notFound } from 'next/navigation'
import { getArticleContent, getProjectStaticParams } from '../../../lib/content'
import matter from 'gray-matter'
import { MDXRemote } from '../../../components/MDXRemote'

export const generateStaticParams = getProjectStaticParams

export default async function ProjectArticle({ params }: {
  params: Promise<{ slug: string }>
}) {
  const { slug } = await params
  const content = getArticleContent('project', null, slug)
  
  if (!content) {
    notFound()
  }
  
  const { content: mdxContent } = matter(content)
  
  return (
    <article className="max-w-4xl mx-auto px-6 py-12 prose">
      <MDXRemote source={mdxContent} />
    </article>
  )
}
