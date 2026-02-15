const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const net = require('net');
const fs = require('fs');
const http = require('http');

let mainWindow;
let serverProcess;
let pythonProcess;

const PYTHON_PORT = 9847;

// ── Helper: find an available port dynamically ───────────────────────────────

async function getFreePort() {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.listen(0, () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

// ── Helper: wait for a port to accept connections ────────────────────────────

function waitForPort(port, host, timeout) {
  host = host || '127.0.0.1';
  timeout = timeout || 30000;
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      const sock = new net.Socket();
      sock.setTimeout(500);
      sock.on('connect', () => {
        sock.destroy();
        resolve();
      });
      sock.on('error', () => {
        sock.destroy();
        if (Date.now() - start > timeout) {
          reject(new Error('Port ' + port + ' not ready after ' + timeout + 'ms'));
        } else {
          setTimeout(check, 300);
        }
      });
      sock.on('timeout', () => {
        sock.destroy();
        setTimeout(check, 300);
      });
      sock.connect(port, host);
    };
    check();
  });
}

// ── Find Python executable ───────────────────────────────────────────────────

function findPython() {
  const { execSync } = require('child_process');
  const candidates = process.platform === 'win32'
    ? ['python', 'python3', 'py']
    : ['python3', 'python'];
  for (const cmd of candidates) {
    try {
      execSync(cmd + ' --version', { stdio: 'pipe' });
      return cmd;
    } catch (e) { /* try next */ }
  }
  return 'python3';
}

// ── Start Python FastAPI backend ─────────────────────────────────────────────

function getBackendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend');
  }
  return path.join(__dirname, '..', 'backend');
}

async function startPythonServer() {
  const backendPath = getBackendPath();
  const serverScript = path.join(backendPath, 'server.py');

  if (!fs.existsSync(serverScript)) {
    console.error('[Python] server.py not found at:', serverScript);
    return;
  }

  const python = findPython();
  console.log('[Python] Starting: ' + python + ' ' + serverScript);

  pythonProcess = spawn(python, [serverScript, '--port', String(PYTHON_PORT)], {
    cwd: backendPath,
    env: Object.assign({}, process.env),
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.on('data', function (data) {
    console.log('[Python] ' + data.toString().trim());
  });

  pythonProcess.stderr.on('data', function (data) {
    // uvicorn logs to stderr
    console.log('[Python] ' + data.toString().trim());
  });

  pythonProcess.on('error', function (err) {
    console.error('[Python] Failed to start:', err);
  });

  pythonProcess.on('exit', function (code) {
    console.log('[Python] Exited with code ' + code);
    pythonProcess = null;
  });

  try {
    await waitForPort(PYTHON_PORT, '127.0.0.1', 20000);
    console.log('[Python] Server ready on port ' + PYTHON_PORT);
  } catch (err) {
    console.error('[Python] Server did not start in time:', err.message);
  }
}

// ── Start Next.js server (production only) ───────────────────────────────────

async function startNextServer(port) {
  const isDev = process.env.NODE_ENV === 'development';

  const serverPath = isDev
    ? path.join(__dirname, '..', '.next/standalone/server.js')
    : path.join(process.resourcesPath, 'app/server.js');

  serverProcess = spawn(process.execPath, [serverPath], {
    env: {
      ...process.env,
      PORT: port,
      NODE_ENV: 'production',
      ELECTRON_RUN_AS_NODE: '1',
    },
  });

  serverProcess.stdout.on('data', (data) => console.log('Next.js: ' + data));
  serverProcess.stderr.on('data', (data) => console.error('Next.js Error: ' + data));
}

// ── Create main window ──────────────────────────────────────────────────────

async function createWindow() {
  const isDev = process.env.NODE_ENV === 'development';
  const nextPort = isDev ? 3000 : await getFreePort();

  // Start Python backend first
  await startPythonServer();

  // Start Next.js in production
  if (!isDev) {
    await startNextServer(nextPort);
  }

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    backgroundColor: '#09090b',
    title: 'Paper2Notebook',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    icon: path.join(__dirname, '..', 'public', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const url = 'http://127.0.0.1:' + nextPort;

  // Retry logic to wait for the server to boot
  const loadURL = () => {
    mainWindow.loadURL(url).catch(() => {
      setTimeout(loadURL, 200);
    });
  };

  loadURL();

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    // Tell renderer that Python server is up
    mainWindow.webContents.send('python-server-ready', PYTHON_PORT);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── IPC Handlers ─────────────────────────────────────────────────────────────

ipcMain.handle('get-python-port', () => PYTHON_PORT);

ipcMain.handle('open-file-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'PDF Files', extensions: ['pdf'] }],
  });
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('open-external', async (_event, url) => {
  shell.openExternal(url);
});

ipcMain.handle('save-notebook', async (_event, sessionId) => {
  const result = await dialog.showSaveDialog({
    defaultPath: 'notebook.ipynb',
    filters: [{ name: 'Jupyter Notebook', extensions: ['ipynb'] }],
  });
  if (!result.canceled && result.filePath) {
    try {
      return new Promise((resolve, reject) => {
        http.get('http://127.0.0.1:' + PYTHON_PORT + '/api/notebook/' + sessionId, (res) => {
          const chunks = [];
          res.on('data', (chunk) => chunks.push(chunk));
          res.on('end', () => {
            fs.writeFileSync(result.filePath, Buffer.concat(chunks));
            resolve(result.filePath);
          });
        }).on('error', reject);
      });
    } catch (e) {
      return null;
    }
  }
  return null;
});

// ── App lifecycle ────────────────────────────────────────────────────────────

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (serverProcess) serverProcess.kill();
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('quit', () => {
  if (serverProcess) serverProcess.kill();
  if (pythonProcess) pythonProcess.kill();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});