import { MDXRemote as MDXRemoteBase } from 'next-mdx-remote/rsc'

export function MDXRemote({ source }: {
  source: string
}) {
  return <MDXRemoteBase source={source} />
}
