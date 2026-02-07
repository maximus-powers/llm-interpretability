import { MDXRemote as MDXRemoteBase } from 'next-mdx-remote/rsc'
import remarkGfm from 'remark-gfm'

export function MDXRemote({ source }: {
  source: string
}) {
  return (
    <MDXRemoteBase 
      source={source} 
      options={{
        mdxOptions: {
          remarkPlugins: [remarkGfm],
        },
      }}
    />
  )
}
