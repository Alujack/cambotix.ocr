// The CommonJS build resolves modules without package "exports", so it cannot
// see this subpath's own declarations; its API is the main entry's.
declare module "onnxruntime-web/wasm" {
  export * from "onnxruntime-web";
}
