#!/usr/bin/env node
// Copies the browser runtime files @khmerocr/local needs into your site, so
// they are served by you instead of a CDN:
//
//   npx khmerocr-local-assets public/khmerocr
//
// then load with
//   onnx: { wasmPaths: "/khmerocr/ort/" },
//   tesseractWorker: { workerPath: "/khmerocr/tesseract/worker.min.js", corePath: "/khmerocr/tesseract/" }
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";

const target = process.argv[2];
if (!target || process.argv.includes("--help")) {
  console.error("usage: khmerocr-local-assets <folder>   e.g. khmerocr-local-assets public/khmerocr");
  process.exit(target ? 0 : 1);
}
const require = createRequire(import.meta.url);
/** A dependency's folder (not every package exports its package.json). */
function packageDir(name) {
  let dir = dirname(require.resolve(name));
  while (!existsSync(join(dir, "package.json")) || JSON.parse(readFileSync(join(dir, "package.json"), "utf8")).name !== name) {
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`cannot find the ${name} package`);
    dir = parent;
  }
  return dir;
}

const groups = [
  ["ort", join(packageDir("onnxruntime-web"), "dist"), ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]],
  ["tesseract", join(packageDir("tesseract.js"), "dist"), ["worker.min.js"]],
  // tesseract.js picks one of these by the browser's WebAssembly features (LSTM-only engine).
  ["tesseract", packageDir("tesseract.js-core"),
    ["tesseract-core-lstm.wasm.js", "tesseract-core-simd-lstm.wasm.js", "tesseract-core-relaxedsimd-lstm.wasm.js"]],
];
for (const [folder, from, files] of groups) {
  const to = resolve(target, folder);
  mkdirSync(to, { recursive: true });
  for (const file of files) copyFileSync(join(from, file), join(to, file));
  console.log(`${files.length} file(s) -> ${to}`);
}
