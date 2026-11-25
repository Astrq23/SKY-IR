const POINTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/** Compass bearing in degrees (clockwise from north) -> "NE 47°". */
export function compass(bearing: number): string {
  const b = ((bearing % 360) + 360) % 360;
  return `${POINTS[Math.round(b / 45) % 8]} ${b.toFixed(0)}°`;
}

/** Camera-relative heading (0 = away from the camera) -> compass bearing, given the camera's yaw. */
export const toBearing = (heading: number, yawDeg: number) => (((heading + yawDeg) % 360) + 360) % 360;
