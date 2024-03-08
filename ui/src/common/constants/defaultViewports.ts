import { Viewport } from '../classes/Viewport';
import { ViewportId } from '../enums/ViewportId';

export const defaultViewports = [
   new Viewport(ViewportId.xs, 0, 543),
   new Viewport(ViewportId.sm, 544, 767),
   new Viewport(ViewportId.md, 768, 1023),
   new Viewport(ViewportId.lg, 1024, 1279),
   new Viewport(ViewportId.xl, 1280, Number.MAX_SAFE_INTEGER),
]