import { dbClient } from './common/constants/dbClient.js';
import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { processBoxScores } from './common/functions/processBoxScores.js';

await dbClient.connect();
createEndpoints(initialize());
//const browser = await puppeteer.launch({ headless: false });
//const page = await browser.newPage();
//await retrieveWebSchedules(page);
//await retrieveWebBoxscores(page);
await processBoxScores();