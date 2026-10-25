export interface ComponentStats {
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
}

/**
 * OpenCV `connectedComponentsWithStats(mask, connectivity=8)` for a 0/1 mask.
 * Returns the label image (0 is the background) and the stats of labels
 * 1..n. Labels are numbered in raster order of each component's first pixel.
 */
export function connectedComponents(mask: Uint8Array, width: number, height: number):
  { labels: Int32Array; stats: ComponentStats[] } {
  const labels = new Int32Array(width * height);
  const parent: number[] = [0];
  const find = (a: number): number => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]!]!;
      a = parent[a]!;
    }
    return a;
  };
  const union = (a: number, b: number): number => {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return ra;
    const root = Math.min(ra, rb);
    parent[Math.max(ra, rb)] = root;
    return root;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!mask[i]) continue;
      let label = 0;
      const neighbours = [
        x > 0 ? labels[i - 1]! : 0,
        y > 0 && x > 0 ? labels[i - width - 1]! : 0,
        y > 0 ? labels[i - width]! : 0,
        y > 0 && x + 1 < width ? labels[i - width + 1]! : 0,
      ];
      for (const n of neighbours) {
        if (n === 0) continue;
        label = label === 0 ? find(n) : union(label, n);
      }
      if (label === 0) {
        label = parent.length;
        parent.push(label);
      }
      labels[i] = label;
    }
  }
  // Relabel roots densely in order of first appearance.
  const dense = new Map<number, number>();
  const stats: ComponentStats[] = [];
  const bounds: number[][] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!labels[i]) continue;
      const root = find(labels[i]!);
      let id = dense.get(root);
      if (id === undefined) {
        id = dense.size + 1;
        dense.set(root, id);
        bounds.push([x, y, x, y, 0]);
      }
      labels[i] = id;
      const b = bounds[id - 1]!;
      b[0] = Math.min(b[0]!, x);
      b[1] = Math.min(b[1]!, y);
      b[2] = Math.max(b[2]!, x);
      b[3] = Math.max(b[3]!, y);
      b[4]!++;
    }
  }
  for (const [x0, y0, x1, y1, area] of bounds) {
    stats.push({ x: x0!, y: y0!, width: x1! - x0! + 1, height: y1! - y0! + 1, area: area! });
  }
  return { labels, stats };
}
