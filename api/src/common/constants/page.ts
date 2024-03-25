import { Milliseconds } from '../enums/Milliseconds.js';
import { browser } from './browser.js';

const pageInit = await browser.newPage();
pageInit.setDefaultNavigationTimeout(Milliseconds.minute);
pageInit.setDefaultTimeout(Milliseconds.minute);
await pageInit.setRequestInterception(true);
pageInit.on('request', request => {
   (async () => {
      if (request.resourceType() === 'image' || request.resourceType() === 'font') {
         await request.abort();
      } else {
         await request.continue();
      }
   })()
});

export const page = pageInit;