import puppeteer from 'puppeteer';
import { dbClient } from './common/constants/dbClient.js';
import { Milliseconds } from './common/enums/Milliseconds.js';
import { createEndpoints } from './common/functions/createEndpoints.js';
import { initialize } from './common/functions/initialize.js';
import { processBoxScores } from './common/functions/processBoxScores.js';
import { retrieveWebSchedules } from './common/functions/retrieveWebSchedules.js';

await dbClient.connect();
createEndpoints(initialize());
const browser = await puppeteer.launch();
const page = await browser.newPage();
await page.setDefaultNavigationTimeout(5 * Milliseconds.minute);
await page.setDefaultTimeout(5 * Milliseconds.minute);
await page.setRequestInterception(true);
page.on('request', request => {
   (async () => {
      if (request.resourceType() === 'image') {
         await request.abort();
      } else {
         await request.continue();
      }
   })()
});
await retrieveWebSchedules(page);
//await retrieveWebBoxscores(page);
await processBoxScores(page);
