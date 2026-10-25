export const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01]);
export const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d]);
export const WEBP = new Uint8Array([...new TextEncoder().encode("RIFF"), 0x24, 0x00, 0x00, 0x00,
  ...new TextEncoder().encode("WEBPVP8 ")]);

export interface Call {
  url: string;
  init: RequestInit;
}

/** A fake fetch that records calls and answers with `respond`. */
export function fakeFetch(respond: (call: Call) => Response | Promise<Response> = () => json({})) {
  const calls: Call[] = [];
  const fetch = async (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> => {
    const call = { url: String(input), init };
    calls.push(call);
    return respond(call);
  };
  return { fetch: fetch as typeof globalThis.fetch, calls };
}

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "x-request-id": "req_server", ...headers },
  });
}

export function header(call: Call, name: string): string | null {
  return new Headers(call.init.headers).get(name);
}

export const DOCUMENT = {
  schema_version: "1.0",
  document_type: "khmer_national_id",
  status: "review_required",
  fields: {
    id_number: { value: "000000001", confidence: 0.97, source: "front_region", valid: true },
    full_name_km: { value: null, confidence: null, source: "front_region", valid: false },
    full_name_latin: { value: "KUMRU SAKALBANG", confidence: 0.95, source: "front_region", valid: true },
    date_of_birth: { value: "1990-01-01", confidence: 0.6, source: "front_region", valid: true },
    gender: { value: "M", confidence: 0.9, source: "front_region", valid: true },
  },
  warnings: ["layout_template_unverified", "low_confidence:date_of_birth", "field_not_detected:full_name_km"],
  review_required: true,
  metadata: { template_id: "khmer_national_id_front_v1" },
} as const;
