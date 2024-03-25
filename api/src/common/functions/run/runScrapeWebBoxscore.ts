import { initializeChildProcess } from '../initializeChildProcess.js';
import { scrapeWebBoxscore } from '../scrapeWebBoxscore.js';

await initializeChildProcess();
await scrapeWebBoxscore();