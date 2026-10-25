import { it } from "vitest";
import { resize } from "../src/image/resize.js";
import { decode, fixtures } from "./helpers.js";
it("diag", async () => {
  const gray = await decode(fixtures["gray"]);
  const e = await decode(fixtures["resize"]["area_up"]);
  const a = resize(gray, e.width, e.height, "area");
  const rows = new Map<number, number>(), cols = new Map<number, number>();
  for (let y = 0; y < e.height; y++) for (let x = 0; x < e.width; x++) {
    const d = Math.abs(a.data[y * e.width + x]! - e.data[y * e.width + x]!);
    if (d > 1) { rows.set(y, (rows.get(y) ?? 0) + 1); cols.set(x, (cols.get(x) ?? 0) + 1); }
  }
  console.log("rows", [...rows].slice(0, 20), "cols", [...cols].slice(0, 20));
});
