// Class colours tuned for contrast against grey / ironbow thermal imagery.
export const CLASS_COLORS = ["#ffd166", "#4cc9f0", "#7bf1a8", "#f472d0", "#9aa4b2"];
export const LOCK_COLOR = "#ff4d4d";

export const classColor = (cls: number) => CLASS_COLORS[cls % CLASS_COLORS.length];

export const PALETTE_LABELS: Record<string, string> = {
  "white-hot": "White-hot",
  "black-hot": "Black-hot",
  ironbow: "Ironbow",
  arctic: "Arctic",
  rainbow: "Rainbow",
};
