import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  // Original
  sendNotification: (text: string) => ipcRenderer.send('notify', text),
  getAppVersion: () => process.versions.electron,
  // Pipeline
  getPythonPort: () => ipcRenderer.invoke('get-python-port'),
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  openExternal: (url: string) => ipcRenderer.invoke('open-external', url),
  saveNotebook: (sessionId: string) => ipcRenderer.invoke('save-notebook', sessionId),
  onPythonReady: (callback: (port: number) => void) =>
    ipcRenderer.on('python-server-ready', (_event, port) => callback(port)),
});