import type { Result } from '../interfaces/Result.js';

export const output = (result: Result) => console.log(JSON.stringify(result));