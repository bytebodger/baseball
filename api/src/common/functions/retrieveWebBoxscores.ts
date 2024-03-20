import dayjs from 'dayjs';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import { getOldestUnretrievedBoxscore } from './queries/getOldestUnretrievedBoxscore.js';
import { updateWebBoxscore } from './queries/updateWebBoxscore.js';
import { wait } from './wait.js';

export const retrieveWebBoxscores = async (page: Page): Promise<void> => {
   const { rows: boxscores } = await getOldestUnretrievedBoxscore() as { rows: WebBoxscoreTable[] };
   if (!boxscores.length)
      return;
   const { url, web_boxscore_id } = boxscores[0];
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   await page.waitForSelector('#event_1');
   const html = await page.content();
   console.log('updating web boxscore', {
      time_retrieved: dayjs().utc().unix(),
      web_boxscore_id,
   })
   await updateWebBoxscore({
      html,
      time_retrieved: dayjs().utc().unix(),
      web_boxscore_id,
   })
   await wait(4 * Milliseconds.second);
   return await retrieveWebBoxscores(page);
}