import dayjs from 'dayjs';
import { page } from '../constants/page.js';
import type { Result } from '../interfaces/Result.js';
import type { WebBoxscoreTable } from '../interfaces/tables/WebBoxscoreTable.js';
import { output } from './output.js';
import { getDBOldestUnretrievedBoxscore } from './queries/getDBOldestUnretrievedBoxscore.js';
import { updateDBWebBoxscore } from './queries/updateDBWebBoxscore.js';

export const scrapeWebBoxscore = async () => {
   const result: Result = {
      errors: [],
      function: 'scrapeWebBoxscore()',
      messages: [],
      proceed: false,
   };
   const { rows: boxscores } = await getDBOldestUnretrievedBoxscore() as { rows: WebBoxscoreTable[] };
   if (!boxscores.length) {
      result.messages.push('There are no unretrieved boxscores.');
      return output(result);
   }
   const { url, web_boxscore_id } = boxscores[0];
   await page.goto(url, { waitUntil: 'domcontentloaded' });
   await page.waitForSelector('#event_1');
   const html = await page.content();
   const fields = {
      time_retrieved: dayjs().utc().unix(),
      web_boxscore_id,
   };
   await updateDBWebBoxscore({
      html,
      ...fields,
   })
   result.messages.push('updated web boxscore:');
   result.messages.push(fields);
   result.proceed = true;
   return output(result);
}