import type { GenericObject } from '../types/GenericObject.js';

export interface Result {
   errors: Array<string | GenericObject>,
   function: string,
   messages: Array<string | GenericObject>,
   proceed: boolean,
}