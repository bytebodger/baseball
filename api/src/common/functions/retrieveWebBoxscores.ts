import dayjs from 'dayjs';
import type { Page } from 'puppeteer';
import { Milliseconds } from '../enums/Milliseconds.js';
import type { WebBoxscore } from '../interfaces/tables/WebBoxscore.js';
import { getOldestUnretrievedBoxscore } from './queries/getOldestUnretrievedBoxscore.js';
import { updateWebBoxscore } from './queries/updateWebBoxscore.js';
import { sleep } from './sleep.js';

export const retrieveWebBoxscores = async (page: Page): Promise<void> => {
   const { rows: boxscores } = await getOldestUnretrievedBoxscore() as { rows: WebBoxscore[] };
   if (!boxscores.length)
      return;
   const { url, web_boxscore_id } = boxscores[0];
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   await page.waitForSelector('#event_1');
   const html = await page.content();
   await updateWebBoxscore({
      html,
      time_retrieved: dayjs().utc().valueOf(),
      web_boxscore_id,
   })
   await sleep(4 * Milliseconds.second);
   return await retrieveWebBoxscores(page);
}