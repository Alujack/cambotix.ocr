/**
 * Route each region to the recognizer suited to its script. A route may list
 * several backends: the first answers `recognizeLine`, and
 * `recognizeCandidates` yields each in turn so the extractor can fall back
 * when a read does not validate.
 */
import { ocrError } from "../errors.js";
import type { Raster } from "../image/raster.js";
import type { TextRegion } from "../schemas.js";
import { DEFAULT_HINT, Recognizer, type RecognizedText, type RegionHint, type Script } from "./base.js";

export class ScriptRouter extends Recognizer {
  readonly name = "router";
  readonly backends: Map<string, Recognizer>;
  readonly routes: Map<string, string[]>;
  readonly defaultBackend: string;
  readonly page: string;

  constructor(backends: Map<string, Recognizer>, routes: Record<string, string | string[]>,
              defaultBackend: string, page?: string | null) {
    super();
    this.routes = new Map(Object.entries(routes).map(([script, names]) =>
      [script, typeof names === "string" ? [names] : [...names]]));
    const referenced = new Set([defaultBackend, page ?? defaultBackend]);
    for (const names of this.routes.values()) {
      if (names.length === 0) throw ocrError("MODEL_NOT_AVAILABLE", "A route must list at least one backend.");
      names.forEach((name) => referenced.add(name));
    }
    const unknown = [...referenced].filter((name) => !backends.has(name)).sort();
    if (unknown.length > 0) {
      throw ocrError("MODEL_NOT_AVAILABLE", "Routes reference unknown backends.", { unknown });
    }
    this.backends = backends;
    this.defaultBackend = defaultBackend;
    this.page = page ?? defaultBackend;
  }

  backendsFor(script: Script): Recognizer[] {
    return (this.routes.get(script) ?? [this.defaultBackend]).map((name) => this.backends.get(name)!);
  }

  recognizeLine(image: Raster, hint: RegionHint = DEFAULT_HINT): Promise<RecognizedText> {
    return this.backendsFor(hint.script)[0]!.recognizeLine(image, hint);
  }

  override async *recognizeCandidates(image: Raster, hint: RegionHint = DEFAULT_HINT):
    AsyncGenerator<RecognizedText> {
    for (const backend of this.backendsFor(hint.script)) {
      yield await backend.recognizeLine(image, hint);
    }
  }

  override recognizePage(image: Raster): Promise<TextRegion[]> {
    return this.backends.get(this.page)!.recognizePage(image);
  }

  override describe(): Record<string, string> {
    const description: Record<string, string> = { backend: this.name, default: this.defaultBackend };
    for (const [script, names] of [...this.routes].sort(([a], [b]) => a.localeCompare(b))) {
      description[`route.${script}`] = names.join(",");
    }
    for (const [name, recognizer] of [...this.backends].sort(([a], [b]) => a.localeCompare(b))) {
      description[`backends.${name}`] = recognizer.describe()["backend"] ?? name;
    }
    return description;
  }

  override async close(): Promise<void> {
    await Promise.all([...this.backends.values()].map((backend) => backend.close()));
  }
}
