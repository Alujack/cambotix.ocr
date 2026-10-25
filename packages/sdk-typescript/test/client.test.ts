import { describe, expect, it } from "vitest";

import { KhmerOCRClient, KhmerOCRError } from "../src/index.js";
import { DOCUMENT, JPEG, PNG, WEBP, fakeFetch, header, json, type Call } from "./helpers.js";

function client(respond?: (call: Call) => Response | Promise<Response>, options: Record<string, unknown> = {}) {
  const fake = fakeFetch(respond);
  return { ocr: new KhmerOCRClient({ baseUrl: "http://10.0.0.5:8000", fetch: fake.fetch, ...options }), ...fake };
}

async function caught(promise: Promise<unknown>): Promise<KhmerOCRError> {
  try {
    await promise;
  } catch (error) {
    expect(error).toBeInstanceOf(KhmerOCRError);
    return error as KhmerOCRError;
  }
  throw new Error("expected the call to fail");
}

describe("base URL", () => {
  it.each([
    ["http://10.0.0.5:8000", "http://10.0.0.5:8000/v1/health"],
    ["http://10.0.0.5:8000/", "http://10.0.0.5:8000/v1/health"],
    ["http://10.0.0.5:8000/v1", "http://10.0.0.5:8000/v1/health"],
    ["https://ocr.example.com/v1//", "https://ocr.example.com/v1/health"],
    ["https://example.com/khmerocr", "https://example.com/khmerocr/v1/health"],
  ])("%s", async (baseUrl, expected) => {
    const { fetch, calls } = fakeFetch(() => json({ status: "ok" }));
    await new KhmerOCRClient({ baseUrl, fetch }).health();
    expect(calls[0]?.url).toBe(expected);
  });

  it("requires a base URL", () => {
    expect(() => new KhmerOCRClient({ baseUrl: "" })).toThrow(TypeError);
  });
});

describe("headers", () => {
  it("sends the API key, extra headers and request ID", async () => {
    const { ocr, calls } = client(() => json({ status: "ok" }), {
      apiKey: "secret-key-0123456789", headers: { "X-App-Token": "user-token" },
    });
    await ocr.health({ requestId: "scan-42" });
    const call = calls[0]!;
    expect(call.init.method).toBe("GET");
    expect(header(call, "authorization")).toBe("Bearer secret-key-0123456789");
    expect(header(call, "x-app-token")).toBe("user-token");
    expect(header(call, "x-request-id")).toBe("scan-42");
  });

  it("omits Authorization without a key", async () => {
    const { ocr, calls } = client(() => json({ status: "ok" }));
    await ocr.health();
    expect(header(calls[0]!, "authorization")).toBeNull();
    expect(header(calls[0]!, "x-request-id")).toBeNull();
  });
});

describe("uploads", () => {
  it("posts the card as multipart with include_text_regions", async () => {
    const { ocr, calls } = client(() => json(DOCUMENT));
    const result = await ocr.extractNationalId(JPEG, { includeTextRegions: true, filename: "front.jpg" });
    const call = calls[0]!;
    expect(call.url).toBe("http://10.0.0.5:8000/v1/documents/national-id/extract");
    expect(call.init.method).toBe("POST");
    const form = call.init.body as FormData;
    const file = form.get("file") as File;
    expect(file.name).toBe("front.jpg");
    expect(file.type).toBe("image/jpeg");
    expect(new Uint8Array(await file.arrayBuffer())).toEqual(JPEG);
    expect(form.get("include_text_regions")).toBe("true");
    // fetch must set the multipart boundary itself.
    expect(header(call, "content-type")).toBeNull();
    expect(result.fields.id_number?.value).toBe("000000001");
  });

  it("defaults include_text_regions to false on extract endpoints", async () => {
    const { ocr, calls } = client(() => json({ ...DOCUMENT, document_type: "student_id_card" }));
    await ocr.extractStudentId(PNG);
    expect(calls[0]?.url).toBe("http://10.0.0.5:8000/v1/documents/student-id/extract");
    expect((calls[0]!.init.body as FormData).get("include_text_regions")).toBe("false");
  });

  it.each([
    ["checkQuality", "/v1/documents/quality-check"],
    ["recognizeText", "/v1/ocr"],
  ] as const)("%s sends only the file", async (method, path) => {
    const { ocr, calls } = client(() => json({}));
    await ocr[method](WEBP);
    expect(calls[0]?.url).toBe(`http://10.0.0.5:8000${path}`);
    const form = calls[0]!.init.body as FormData;
    expect([...form.keys()]).toEqual(["file"]);
    expect((form.get("file") as File).type).toBe("image/webp");
  });

  it("sniffs bytes and names the upload by type", async () => {
    for (const [bytes, type, name] of [[JPEG, "image/jpeg", "image.jpg"], [PNG, "image/png", "image.png"],
      [WEBP, "image/webp", "image.webp"]] as const) {
      const { ocr, calls } = client(() => json({}));
      await ocr.checkQuality(bytes.buffer);
      const file = (calls[0]!.init.body as FormData).get("file") as File;
      expect([file.type, file.name]).toEqual([type, name]);
    }
  });

  it("accepts a Node Buffer view with an offset", async () => {
    const pooled = Buffer.concat([Buffer.from([1, 2, 3]), Buffer.from(PNG)]).subarray(3);
    const { ocr, calls } = client(() => json({}));
    await ocr.checkQuality(pooled);
    const file = (calls[0]!.init.body as FormData).get("file") as File;
    expect(file.type).toBe("image/png");
    expect(new Uint8Array(await file.arrayBuffer())).toEqual(PNG);
  });

  it("uses a File's own name and type", async () => {
    const { ocr, calls } = client(() => json({}));
    await ocr.checkQuality(new File([JPEG], "camera.jpg", { type: "image/jpeg" }));
    const file = (calls[0]!.init.body as FormData).get("file") as File;
    expect([file.name, file.type]).toEqual(["camera.jpg", "image/jpeg"]);
  });

  it("sniffs a Blob whose declared type is missing or generic", async () => {
    const { ocr, calls } = client(() => json({}));
    await ocr.checkQuality(new Blob([PNG], { type: "application/octet-stream" }));
    expect(((calls[0]!.init.body as FormData).get("file") as File).type).toBe("image/png");
  });

  it("lets an explicit content type win", async () => {
    const { ocr, calls } = client(() => json({}));
    await ocr.checkQuality(new Uint8Array([1, 2, 3]), { contentType: "image/png" });
    expect(((calls[0]!.init.body as FormData).get("file") as File).type).toBe("image/png");
  });

  it.each([
    ["unknown bytes", new Uint8Array([0x25, 0x50, 0x44, 0x46]), {}, "UNSUPPORTED_MEDIA_TYPE"],
    ["unsupported explicit type", JPEG, { contentType: "image/gif" }, "UNSUPPORTED_MEDIA_TYPE"],
    ["empty", new Uint8Array(), {}, "INVALID_IMAGE"],
    ["empty blob", new Blob([]), {}, "INVALID_IMAGE"],
  ])("rejects %s before any request", async (_name, image, options, code) => {
    const { ocr, calls } = client();
    const error = await caught(ocr.extractNationalId(image, options));
    expect(error.code).toBe(code);
    expect(error.status).toBeNull();
    expect(calls).toHaveLength(0);
  });

  it("rejects images over maxUploadBytes before any request", async () => {
    const { ocr, calls } = client(undefined, { maxUploadBytes: 8 });
    const error = await caught(ocr.extractNationalId(JPEG));
    expect(error.code).toBe("PAYLOAD_TOO_LARGE");
    expect(error.details).toEqual({ size: JPEG.length, max_upload_bytes: 8 });
    expect(calls).toHaveLength(0);
  });
});

describe("errors", () => {
  it("maps the server error envelope", async () => {
    const { ocr } = client(() => json({
      error: { code: "IMAGE_QUALITY_TOO_LOW", message: "Too blurry.", request_id: "req_body",
        details: { issues: ["too_blurry"] } },
    }, 422));
    const error = await caught(ocr.extractNationalId(JPEG));
    expect(error).toMatchObject({ code: "IMAGE_QUALITY_TOO_LOW", message: "Too blurry.", status: 422,
      requestId: "req_body", details: { issues: ["too_blurry"] } });
  });

  it("maps 401 UNAUTHORIZED and falls back to the X-Request-ID header", async () => {
    const { ocr } = client(() => json({ error: { code: "UNAUTHORIZED", message: "A valid API key is required." } },
      401, { "x-request-id": "req_header" }));
    const error = await caught(ocr.models());
    expect(error).toMatchObject({ code: "UNAUTHORIZED", status: 401, requestId: "req_header", details: null });
  });

  it("reports non-envelope error bodies as UNEXPECTED_RESPONSE", async () => {
    const { ocr } = client(() => new Response("<html>Bad gateway</html>", { status: 502 }));
    const error = await caught(ocr.health());
    expect(error).toMatchObject({ code: "UNEXPECTED_RESPONSE", status: 502 });
  });

  it("reports a 2xx body that is not JSON as UNEXPECTED_RESPONSE", async () => {
    const { ocr } = client(() => new Response("ok", { status: 200 }));
    expect((await caught(ocr.health())).code).toBe("UNEXPECTED_RESPONSE");
  });

  it("reports transport failures as NETWORK_ERROR", async () => {
    const { ocr } = client(() => Promise.reject(new TypeError("fetch failed")));
    const error = await caught(ocr.health());
    expect(error).toMatchObject({ code: "NETWORK_ERROR", status: null });
    expect(error.cause).toBeInstanceOf(TypeError);
  });

  it("times out", async () => {
    const { ocr } = client(hangUntilAborted, { timeoutMs: 20 });
    expect((await caught(ocr.health())).code).toBe("TIMEOUT");
  });

  it("reports a caller abort as ABORTED", async () => {
    const { ocr } = client(hangUntilAborted);
    const controller = new AbortController();
    const pending = ocr.extractNationalId(JPEG, { signal: controller.signal });
    setTimeout(() => controller.abort(), 5);
    expect((await caught(pending)).code).toBe("ABORTED");
  });

  it("does not send a request when the signal is already aborted", async () => {
    const { ocr, calls } = client();
    const error = await caught(ocr.health({ signal: AbortSignal.abort() }));
    expect(error.code).toBe("ABORTED");
    expect(calls).toHaveLength(0);
  });
});

function hangUntilAborted(call: Call): Promise<Response> {
  return new Promise((_resolve, reject) => {
    call.init.signal?.addEventListener("abort", () =>
      reject(new DOMException("The operation was aborted.", "AbortError")));
  });
}
