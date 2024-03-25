import { initializeChildProcess } from '../initializeChildProcess.js';
import { scrapeWebSchedule } from '../scrapeWebSchedule.js';

await initializeChildProcess();
await scrapeWebSchedule();
