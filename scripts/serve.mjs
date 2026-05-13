import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.join(process.cwd(), 'dist');
const port = Number(process.env.PORT || 4321);
const types = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.xml', 'application/xml; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp']
]);

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${port}`);
    let filePath = path.normalize(decodeURIComponent(url.pathname)).replace(/^\/+/, '');
    if (!filePath || url.pathname.endsWith('/')) filePath = path.join(filePath, 'index.html');
    const abs = path.join(root, filePath);
    if (!abs.startsWith(root)) throw new Error('Invalid path');
    const data = await readFile(abs);
    res.setHeader('content-type', types.get(path.extname(abs)) || 'application/octet-stream');
    res.end(data);
  } catch {
    try {
      res.statusCode = 404;
      res.setHeader('content-type', 'text/html; charset=utf-8');
      res.end(await readFile(path.join(root, '404.html')));
    } catch {
      res.statusCode = 404;
      res.end('Not found');
    }
  }
}).listen(port, () => {
  console.log(`MSXF Notes preview: http://localhost:${port}`);
});
