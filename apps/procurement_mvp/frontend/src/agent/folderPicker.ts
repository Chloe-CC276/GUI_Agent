export type PickedExcel = {
  name: string
  file: File
}

export type DirectoryPickResult = {
  folderName: string
  files: PickedExcel[]
}

async function collectExcelFiles(dir: FileSystemDirectoryHandle): Promise<PickedExcel[]> {
  const files: PickedExcel[] = []
  for await (const [, handle] of dir.entries()) {
    if (handle.kind !== 'file') continue
    const name = handle.name
    if (!/\.xlsx?$/i.test(name)) continue
    const file = await handle.getFile()
    files.push({ name, file })
  }
  return files.sort((a, b) => a.name.localeCompare(b.name))
}

/** Prefer Chromium showDirectoryPicker; never invent a path. */
export async function pickExcelDirectory(): Promise<DirectoryPickResult | null> {
  const picker = (window as Window & {
    showDirectoryPicker?: () => Promise<FileSystemDirectoryHandle>
  }).showDirectoryPicker
  if (!picker) {
    return null
  }
  const dir = await picker.call(window)
  const files = await collectExcelFiles(dir)
  return { folderName: dir.name, files }
}
