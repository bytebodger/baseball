export class Viewport {
   public readonly id: number;
   public readonly maxWidth: number;
   public readonly minWidth: number;

   constructor(
      id: number,
      minWidth: number = 0,
      maxWidth: number = Number.MAX_SAFE_INTEGER,
   ) {
      this.id = id;
      this.maxWidth = maxWidth;
      this.minWidth = minWidth;
   }
}