import { initializeChildProcess } from '../initializeChildProcess.js';
import { processBoxScore } from '../processBoxScore.js';

await initializeChildProcess();
await processBoxScore();